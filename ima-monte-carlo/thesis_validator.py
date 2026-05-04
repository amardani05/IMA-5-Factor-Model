"""
Thesis statement validation + display.

Each pitch config gains a required `THESIS_STATEMENT` field. This module
provides quality-bar checks (length, driver count) and a placeholder fallback
when the field is missing — surfacing a warning rather than refusing to run.
"""
from __future__ import annotations

import re
from typing import Optional


PLACEHOLDER = "[Thesis statement not provided]"

DRIVER_MARKERS = [
    "(1)", "(2)", "(3)", "(4)", "(5)",
    "first,", "second,", "third,", "fourth,", "fifth,",
    "1.", "2.", "3.", "4.", "5.",
    "•", "- ",
]


def validate_thesis(thesis: Optional[str]) -> dict:
    """
    Validate a thesis statement.

    Returns:
        {
            "valid": bool,            # passes the minimum quality bar
            "warnings": list[str],
            "word_count": int,
            "driver_count_estimate": int,
            "is_placeholder": bool,
        }

    The function is forgiving: a too-short or too-long thesis still
    returns valid=True so the pipeline keeps running. The warnings are what
    surface in the report. valid=False is reserved for "no thesis at all".
    """
    if not thesis or not thesis.strip():
        return {
            "valid": False,
            "warnings": [
                "Thesis statement is required. Add a THESIS_STATEMENT field to "
                "the pitch config — 30-100 words describing what wins this pitch."
            ],
            "word_count": 0,
            "driver_count_estimate": 0,
            "is_placeholder": True,
        }

    cleaned = thesis.strip()
    is_placeholder = cleaned == PLACEHOLDER or cleaned.startswith("[Thesis")

    words = re.findall(r"\b\w+\b", cleaned)
    word_count = len(words)
    warnings_list: list[str] = []

    if is_placeholder:
        warnings_list.append(
            "THESIS_STATEMENT is a placeholder — replace it with a real thesis."
        )
    elif word_count < 20:
        warnings_list.append(
            f"Thesis is very short ({word_count} words). A good thesis articulates "
            "(1) what wins the pitch and (2) why, in 30-100 words."
        )
    elif word_count > 150:
        warnings_list.append(
            f"Thesis is long ({word_count} words). If you cannot compress to "
            "<100 words, the thesis likely contains background that belongs in "
            "the model rather than the thesis itself."
        )

    # Heuristic driver count
    lower = cleaned.lower()
    driver_count = 0
    for marker in DRIVER_MARKERS:
        if marker.lower() in lower:
            driver_count += lower.count(marker.lower())
    if driver_count == 0:
        # Fall back to counting major conjunctions between clauses
        driver_count = lower.count(" and ") + 1

    if driver_count > 3:
        warnings_list.append(
            f"Thesis appears to have ~{driver_count} drivers. Strong pitches "
            "usually have 1-2. Consider whether secondary drivers are essential "
            "or noise."
        )

    return {
        "valid": True,
        "warnings": warnings_list,
        "word_count": word_count,
        "driver_count_estimate": driver_count,
        "is_placeholder": is_placeholder,
    }


def get_thesis(mod) -> str:
    """Pull THESIS_STATEMENT from a loaded pitch module, with placeholder fallback."""
    text = getattr(mod, "THESIS_STATEMENT", None)
    if not text or not str(text).strip():
        return PLACEHOLDER
    return str(text).strip()


def wrap_thesis(thesis: str, width: int = 60) -> list[str]:
    """Wrap a thesis statement into display lines of approximately `width` chars."""
    import textwrap
    return textwrap.wrap(thesis, width=width)


__all__ = ["validate_thesis", "get_thesis", "wrap_thesis", "PLACEHOLDER"]
