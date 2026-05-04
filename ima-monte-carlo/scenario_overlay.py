"""
Discrete catalyst overlays.

A catalyst is a named discrete event (FDA approval, contract award, litigation
outcome) that modifies the base valuation additively or multiplicatively.

Each Monte Carlo draw:
    1. compute base fair value from the valuation model
    2. for each catalyst: draw an outcome using its probabilities
    3. apply additive or multiplicative impacts in catalyst order
    4. return the adjusted fair value

Multiple catalysts can be independent (default) or grouped as mutually
exclusive — the same `group` name on two catalysts marks them as exclusive,
which means only one catalyst from that group fires per draw (weights of all
outcomes across catalysts in the group must sum to 1.0).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np


@dataclass
class CatalystOutcome:
    probability: float
    value_impact: float
    impact_type: str = "additive"   # "additive" or "multiplicative"

    def __post_init__(self):
        if self.probability < 0 or self.probability > 1:
            raise ValueError(
                f"Catalyst outcome probability must be in [0,1], got {self.probability}"
            )
        if self.impact_type not in ("additive", "multiplicative"):
            raise ValueError(
                f"impact_type must be 'additive' or 'multiplicative', got "
                f"{self.impact_type!r}"
            )


@dataclass
class Catalyst:
    name: str
    outcomes: dict[str, CatalystOutcome]
    group: str | None = None   # If set, catalysts with the same group are mutually exclusive

    def __post_init__(self):
        total = sum(o.probability for o in self.outcomes.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Catalyst '{self.name}' outcome probabilities must sum to 1.0, "
                f"got {total:.6f}"
            )
        self._labels = list(self.outcomes.keys())
        self._probs = np.array([o.probability for o in self.outcomes.values()])

    def sample_indices(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.choice(len(self._labels), size=n, p=self._probs)

    @property
    def labels(self) -> list[str]:
        return list(self._labels)


def make_catalyst(spec: dict | Catalyst) -> Catalyst:
    """
    Build a Catalyst from a dict spec:

        {
            "name": "FDA Approval",
            "outcomes": {
                "approval":  {"probability": 0.6, "value_impact": 5.0,  "impact_type": "additive"},
                "rejection": {"probability": 0.4, "value_impact": -3.0, "impact_type": "additive"},
            },
            "group": "fda",          # optional
        }
    """
    if isinstance(spec, Catalyst):
        return spec
    if not isinstance(spec, dict):
        raise TypeError(f"Expected dict or Catalyst, got {type(spec).__name__}")

    name = spec["name"]
    group = spec.get("group")
    outcomes = {}
    for label, out in spec["outcomes"].items():
        if isinstance(out, CatalystOutcome):
            outcomes[label] = out
        elif isinstance(out, dict):
            outcomes[label] = CatalystOutcome(
                probability=float(out["probability"]),
                value_impact=float(out["value_impact"]),
                impact_type=out.get("impact_type", "additive"),
            )
        else:
            raise ValueError(f"Outcome {label!r} must be dict or CatalystOutcome")
    return Catalyst(name=name, outcomes=outcomes, group=group)


class CatalystOverlay:
    """
    Manages a collection of catalysts and their application to base fair values.
    """

    def __init__(self, catalysts: Iterable[dict | Catalyst] | None = None):
        self.catalysts: list[Catalyst] = [
            make_catalyst(c) for c in (catalysts or [])
        ]
        # Group catalysts by `group` field for mutually-exclusive handling
        self._groups: dict[str, list[int]] = {}
        for i, cat in enumerate(self.catalysts):
            if cat.group is not None:
                self._groups.setdefault(cat.group, []).append(i)

    def __bool__(self) -> bool:
        return bool(self.catalysts)

    def __len__(self) -> int:
        return len(self.catalysts)

    # ------------------------------------------------------------------ #
    #  Sampling
    # ------------------------------------------------------------------ #
    def sample_outcomes(
        self, n: int, rng: np.random.Generator
    ) -> dict[str, np.ndarray]:
        """
        Returns a dict mapping catalyst name → array of N outcome *indices*
        (index into catalyst.labels).

        Independent catalysts: each draws independently.
        Mutually exclusive catalysts (same group): at most one fires per draw.
        """
        result: dict[str, np.ndarray] = {}

        handled_in_group: set[int] = set()

        for group_name, idx_list in self._groups.items():
            # Flatten outcomes across all catalysts in the group into a single
            # categorical distribution plus a "none" bucket if weights sum < 1.
            entries: list[tuple[int, int, float]] = []  # (cat_idx, outcome_idx, probability)
            total = 0.0
            for ci in idx_list:
                cat = self.catalysts[ci]
                for oi, (_, o) in enumerate(cat.outcomes.items()):
                    entries.append((ci, oi, o.probability))
                    total += o.probability

            if abs(total - 1.0) > 1e-6:
                raise ValueError(
                    f"Mutually-exclusive group '{group_name}' outcome probabilities "
                    f"must sum to 1.0 across all catalysts in the group, got {total:.6f}"
                )

            probs = np.array([p for *_, p in entries])
            choice = rng.choice(len(entries), size=n, p=probs)

            for ci in idx_list:
                # Default: each catalyst in the group gets "no event" represented
                # as a sentinel -1 (interpreted as no-op when applying).
                result[self.catalysts[ci].name] = np.full(n, -1, dtype=int)
                handled_in_group.add(ci)

            for k, (ci, oi, _) in enumerate(entries):
                mask = choice == k
                if mask.any():
                    name = self.catalysts[ci].name
                    arr = result[name]
                    arr[mask] = oi
                    result[name] = arr

        # Independent (non-grouped) catalysts
        for i, cat in enumerate(self.catalysts):
            if i in handled_in_group:
                continue
            result[cat.name] = cat.sample_indices(n, rng)

        return result

    # ------------------------------------------------------------------ #
    #  Application
    # ------------------------------------------------------------------ #
    def apply(
        self,
        base_values: np.ndarray,
        sampled_outcomes: dict[str, np.ndarray],
    ) -> np.ndarray:
        """
        Return the post-catalyst fair values for each draw.

        Rules:
            - Additive impacts are summed, then applied once at the end.
            - Multiplicative impacts are multiplied cumulatively.
            - Applied order: multiplicative first, then additive. (This is the
              conventional ordering — multiplicative impacts scale the "clean"
              valuation, additive impacts add catalyst-specific $/share.)
        """
        if not self.catalysts:
            return base_values

        n = base_values.shape[0]
        add_impact = np.zeros(n, dtype=float)
        mul_factor = np.ones(n, dtype=float)

        for cat in self.catalysts:
            outcome_idx = sampled_outcomes[cat.name]
            outcomes_ordered = list(cat.outcomes.values())
            # Build lookup arrays indexed by outcome_idx (use -1 to mean "no fire")
            for oi, out in enumerate(outcomes_ordered):
                mask = outcome_idx == oi
                if not mask.any():
                    continue
                if out.impact_type == "additive":
                    add_impact[mask] += out.value_impact
                else:  # multiplicative
                    mul_factor[mask] *= out.value_impact

        return base_values * mul_factor + add_impact

    # ------------------------------------------------------------------ #
    #  Reporting helpers
    # ------------------------------------------------------------------ #
    def describe(self) -> str:
        if not self.catalysts:
            return "(no catalysts)"
        lines = []
        for cat in self.catalysts:
            grp = f" [group={cat.group}]" if cat.group else ""
            lines.append(f"{cat.name}{grp}:")
            for label, out in cat.outcomes.items():
                sign = "×" if out.impact_type == "multiplicative" else "+"
                lines.append(
                    f"    {label:<20} p={out.probability:.0%}  "
                    f"{sign}{out.value_impact:+.3g}"
                )
        return "\n".join(lines)

    def outcome_label_arrays(
        self, sampled_outcomes: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        """Convert outcome-index arrays to outcome-label arrays (object dtype)."""
        out: dict[str, np.ndarray] = {}
        for cat in self.catalysts:
            idx = sampled_outcomes[cat.name]
            labels = np.array(cat.labels + ["<none>"], dtype=object)
            # -1 maps to last slot (<none>)
            safe_idx = np.where(idx == -1, len(labels) - 1, idx)
            out[cat.name] = labels[safe_idx]
        return out


__all__ = ["Catalyst", "CatalystOutcome", "CatalystOverlay", "make_catalyst"]
