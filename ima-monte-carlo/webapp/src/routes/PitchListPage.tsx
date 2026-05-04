import { useMemo, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useManifest } from "@/hooks/useManifest";
import { PitchCard } from "@/components/PitchCard";
import { modelTypeLabel } from "@/lib/format";

type SortKey = "newest" | "ticker" | "expected_return" | "risk_reward";

const MODEL_TYPES = ["dcf", "multiples", "sotp", "custom"];
const RETURN_BUCKETS: { id: string; label: string; test: (e: number | null) => boolean }[] = [
  { id: "high", label: ">30% upside", test: (e) => (e ?? -Infinity) > 0.30 },
  { id: "mid", label: "0–30%", test: (e) => (e ?? -Infinity) >= 0 && (e ?? -Infinity) <= 0.30 },
  { id: "neg", label: "Negative", test: (e) => (e ?? 0) < 0 },
];

export function PitchListPage() {
  const { manifest, loading, error } = useManifest();
  const [sortKey, setSortKey] = useState<SortKey>("newest");
  const [activeTypes, setActiveTypes] = useState<Set<string>>(new Set());
  const [activeBuckets, setActiveBuckets] = useState<Set<string>>(new Set());
  const [selectMode, setSelectMode] = useState(false);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const navigate = useNavigate();

  const toggleType = (t: string) => {
    setActiveTypes((s) => {
      const n = new Set(s);
      n.has(t) ? n.delete(t) : n.add(t);
      return n;
    });
  };
  const toggleBucket = (b: string) => {
    setActiveBuckets((s) => {
      const n = new Set(s);
      n.has(b) ? n.delete(b) : n.add(b);
      return n;
    });
  };
  const toggleSelected = (id: string) => {
    setSelected((s) => {
      const n = new Set(s);
      n.has(id) ? n.delete(id) : n.add(id);
      return n;
    });
  };

  const pitches = useMemo(() => {
    if (!manifest) return [];
    let p = manifest.pitches.slice();
    if (activeTypes.size > 0) {
      p = p.filter((x) => activeTypes.has((x.model_type || "").toLowerCase()));
    }
    if (activeBuckets.size > 0) {
      p = p.filter((x) =>
        RETURN_BUCKETS.filter((b) => activeBuckets.has(b.id)).some((b) =>
          b.test(x.expected_return),
        ),
      );
    }
    switch (sortKey) {
      case "ticker":
        p.sort((a, b) => a.ticker.localeCompare(b.ticker));
        break;
      case "expected_return":
        p.sort((a, b) => (b.expected_return ?? -Infinity) - (a.expected_return ?? -Infinity));
        break;
      case "risk_reward":
        p.sort((a, b) => (b.risk_reward_ratio ?? -Infinity) - (a.risk_reward_ratio ?? -Infinity));
        break;
      default:
        p.sort((a, b) => (b.generated_at || "").localeCompare(a.generated_at || ""));
    }
    return p;
  }, [manifest, activeTypes, activeBuckets, sortKey]);

  if (loading) {
    return (
      <div className="container">
        <span className="spinner" /> Loading pitches…
      </div>
    );
  }
  if (error || !manifest) {
    return (
      <div className="container empty-state">
        <p>Could not load manifest: {error}</p>
      </div>
    );
  }

  const compareDisabled = selected.size < 2;

  return (
    <div className="container">
      <div className="page-header">
        <div>
          <h1>IMA Monte Carlo — Pitches</h1>
          <p>Probability-weighted valuation analysis for IMA pitches.</p>
        </div>
        <div style={{ display: "flex", gap: 8 }}>
          {!selectMode ? (
            <button
              className="chip"
              onClick={() => {
                setSelectMode(true);
                setSelected(new Set());
              }}
            >
              Compare
            </button>
          ) : (
            <>
              <button
                className="chip"
                onClick={() => {
                  setSelectMode(false);
                  setSelected(new Set());
                }}
              >
                Cancel
              </button>
              <button
                className={`chip ${compareDisabled ? "muted" : ""}`}
                disabled={compareDisabled}
                onClick={() => {
                  navigate(`/compare?pitches=${[...selected].join(",")}`);
                }}
                style={{
                  background: compareDisabled ? undefined : "var(--ink)",
                  color: compareDisabled ? undefined : "white",
                  border: compareDisabled ? undefined : "1px solid var(--ink)",
                }}
              >
                Compare {selected.size} pitch{selected.size === 1 ? "" : "es"}
              </button>
            </>
          )}
        </div>
      </div>

      {pitches.length === 0 && manifest.pitches.length === 0 && (
        <div className="empty-state">
          <p>No published pitches yet.</p>
          <p>
            Run <code>python main.py --pitch your_pitch.py</code> to generate one,
            then push to git.
          </p>
        </div>
      )}

      {manifest.pitches.length > 0 && (
        <>
          <div className="toolbar">
            <label>Sort:</label>
            <select value={sortKey} onChange={(e) => setSortKey(e.target.value as SortKey)}>
              <option value="newest">Newest first</option>
              <option value="ticker">Ticker A→Z</option>
              <option value="expected_return">Expected return (high→low)</option>
              <option value="risk_reward">Risk/Reward (high→low)</option>
            </select>
            <span style={{ width: 8 }} />
            {MODEL_TYPES.map((t) => (
              <button
                key={t}
                className={`chip ${activeTypes.has(t) ? "" : "muted"}`}
                onClick={() => toggleType(t)}
              >
                {modelTypeLabel(t)}
              </button>
            ))}
            <span style={{ width: 8 }} />
            {RETURN_BUCKETS.map((b) => (
              <button
                key={b.id}
                className={`chip ${activeBuckets.has(b.id) ? "" : "muted"}`}
                onClick={() => toggleBucket(b.id)}
              >
                {b.label}
              </button>
            ))}
          </div>

          {pitches.length === 0 ? (
            <div className="empty-state">
              <p>No pitches match the current filters.</p>
            </div>
          ) : (
            <div className="grid-cards">
              {pitches.map((p) => (
                <PitchCard
                  key={p.pitch_id}
                  pitch={p}
                  selectMode={selectMode}
                  selected={selected.has(p.pitch_id)}
                  onToggleSelect={toggleSelected}
                />
              ))}
            </div>
          )}
        </>
      )}

      <p style={{ marginTop: 24, fontSize: 13, color: "var(--text-muted)" }}>
        <Link to="/about">About the methodology →</Link>
      </p>
    </div>
  );
}
