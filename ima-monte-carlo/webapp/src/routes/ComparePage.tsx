import { useEffect, useMemo, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { useManifest } from "@/hooks/useManifest";
import { PitchCard } from "@/components/PitchCard";
import type { PitchData } from "@/data/types";
import {
  fmtCurrency,
  fmtMultiplier,
  fmtPercent,
  fmtPrice,
  modelTypeLabel,
} from "@/lib/format";

function useMultiplePitchData(ids: string[]) {
  const [data, setData] = useState<Record<string, PitchData>>({});
  const [loading, setLoading] = useState(true);
  const [errors, setErrors] = useState<Record<string, string>>({});

  useEffect(() => {
    if (ids.length === 0) {
      setLoading(false);
      return;
    }
    let cancelled = false;
    setLoading(true);
    const next: Record<string, PitchData> = {};
    const errs: Record<string, string> = {};
    Promise.all(
      ids.map((id) =>
        fetch(`${import.meta.env.BASE_URL}pitches/${id}.json`)
          .then((r) => (r.ok ? r.json() : Promise.reject(`HTTP ${r.status}`)))
          .then((d) => {
            next[id] = d;
          })
          .catch((e) => {
            errs[id] = String(e);
          }),
      ),
    ).then(() => {
      if (!cancelled) {
        setData(next);
        setErrors(errs);
        setLoading(false);
      }
    });
    return () => {
      cancelled = true;
    };
  }, [ids.join(",")]);

  return { data, loading, errors };
}

function deriveBanner(pitches: PitchData[]): string {
  if (pitches.length === 0) return "";
  const hi = pitches.reduce((a, b) =>
    (a.probability_analysis.expected_return ?? -Infinity) >
    (b.probability_analysis.expected_return ?? -Infinity)
      ? a
      : b,
  );
  const rr = pitches.reduce((a, b) =>
    (a.probability_analysis.risk_reward_ratio ?? -Infinity) >
    (b.probability_analysis.risk_reward_ratio ?? -Infinity)
      ? a
      : b,
  );
  const calibration = pitches.reduce((a, b) =>
    countWarnings(a) < countWarnings(b) ? a : b,
  );
  return (
    `${hi.ticker} has highest expected return (${fmtPercent(
      hi.probability_analysis.expected_return,
      0,
      true,
    )}). ${rr.ticker} has best risk/reward (${fmtMultiplier(
      rr.probability_analysis.risk_reward_ratio,
      2,
    )}). ${calibration.ticker} has tightest calibration` +
    (calibration.calibration ? ` (${countWarnings(calibration)} warnings).` : ".")
  );
}

function countWarnings(p: PitchData): number {
  if (!p.calibration) return 0;
  return p.calibration.reduce((s, e) => s + (e.warnings?.length ?? 0), 0);
}

function topInputs(p: PitchData, n = 3): { label: string; share: number }[] {
  if (!p.tornado || p.tornado.length === 0) return [];
  const total = p.tornado.reduce((s, e) => s + Math.max(0, e.variance_share), 0) || 1;
  return p.tornado
    .slice()
    .sort((a, b) => b.variance_share - a.variance_share)
    .slice(0, n)
    .map((e) => ({ label: e.label, share: e.variance_share / total }));
}

export function ComparePage() {
  const [params, setParams] = useSearchParams();
  const initial = (params.get("pitches") || "").split(",").filter(Boolean);
  const { manifest, loading: mLoading } = useManifest();
  const [picked, setPicked] = useState<string[]>(initial);

  // Sync picked → URL
  useEffect(() => {
    if (picked.length > 0) setParams({ pitches: picked.join(",") }, { replace: true });
  }, [picked]);

  const { data, loading, errors } = useMultiplePitchData(picked);

  const ordered = useMemo(
    () => picked.map((id) => data[id]).filter(Boolean) as PitchData[],
    [picked, data],
  );

  const togglePick = (id: string) => {
    setPicked((prev) => {
      if (prev.includes(id)) return prev.filter((x) => x !== id);
      if (prev.length >= 4) return prev;
      return [...prev, id];
    });
  };

  if (picked.length === 0) {
    if (mLoading) return <div className="container"><span className="spinner" /> Loading…</div>;
    return (
      <div className="container">
        <div className="page-header">
          <div>
            <h1>Compare pitches</h1>
            <p>Pick 2 to 4 pitches to compare side-by-side.</p>
          </div>
        </div>
        <div className="grid-cards">
          {(manifest?.pitches ?? []).map((p) => (
            <PitchCard
              key={p.pitch_id}
              pitch={p}
              selectMode
              selected={picked.includes(p.pitch_id)}
              onToggleSelect={togglePick}
            />
          ))}
        </div>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="container"><span className="spinner" /> Loading {picked.join(", ")}…</div>
    );
  }

  return (
    <div className="container">
      <div style={{ marginBottom: 14, fontSize: 13, color: "var(--text-muted)" }}>
        <Link to="/">Pitches</Link> → Compare
      </div>
      <div className="page-header">
        <h1>Comparing {ordered.length} pitches</h1>
        <button
          className="chip"
          onClick={() => navigator.clipboard?.writeText(window.location.href)}
        >
          Copy share link
        </button>
      </div>

      {ordered.length > 0 && <div className="compare-banner">{deriveBanner(ordered)}</div>}

      {Object.entries(errors).map(([id, e]) => (
        <div key={id} className="warning-list">
          <li className="warning">Could not load {id}: {e}</li>
        </div>
      ))}

      <div
        className="compare-grid"
        style={{
          gridTemplateColumns: `repeat(${ordered.length}, minmax(220px, 1fr))`,
        }}
      >
        {ordered.map((p) => {
          const pa = p.probability_analysis;
          const top = topInputs(p);
          const warnings = countWarnings(p);
          return (
            <div key={p.pitch_id} className="card">
              <div style={{ display: "flex", justifyContent: "space-between" }}>
                <Link
                  to={`/pitch/${p.pitch_id}`}
                  style={{
                    fontFamily: "JetBrains Mono",
                    fontSize: 22,
                    fontWeight: 700,
                  }}
                >
                  {p.ticker}
                </Link>
                <button
                  className="chip muted"
                  style={{ fontSize: 11 }}
                  onClick={() => togglePick(p.pitch_id)}
                >
                  Remove
                </button>
              </div>
              <div style={{ fontSize: 12, color: "var(--text-muted)", marginBottom: 10 }}>
                {modelTypeLabel(p.model_type)} • {fmtPrice(p.current_price)}
              </div>
              <table className="compact-table">
                <tbody>
                  <tr>
                    <td>Mean fair value</td>
                    <td className="num">{fmtCurrency(p.distribution.mean)}</td>
                  </tr>
                  <tr>
                    <td>Expected return</td>
                    <td className="num">{fmtPercent(pa.expected_return, 1, true)}</td>
                  </tr>
                  <tr>
                    <td>P(upside)</td>
                    <td className="num">{fmtPercent(pa.probability_upside, 0)}</td>
                  </tr>
                  <tr>
                    <td>Risk/Reward</td>
                    <td className="num">{fmtMultiplier(pa.risk_reward_ratio, 2)}</td>
                  </tr>
                  <tr>
                    <td>5% VaR</td>
                    <td className="num">{fmtCurrency(pa.var_5)}</td>
                  </tr>
                </tbody>
              </table>
              <div style={{ marginTop: 12 }}>
                <div className="label">Top drivers</div>
                {top.length === 0 ? (
                  <div style={{ fontSize: 12, color: "var(--text-muted)" }}>—</div>
                ) : (
                  <ul style={{ paddingLeft: 18, margin: "4px 0 0", fontSize: 12 }}>
                    {top.map((t) => (
                      <li key={t.label}>
                        {t.label} ({fmtPercent(t.share, 0)})
                      </li>
                    ))}
                  </ul>
                )}
              </div>
              <div style={{ marginTop: 10 }}>
                <span className={`chip ${warnings > 0 ? "caution" : "success"}`}>
                  {warnings} calibration warning{warnings === 1 ? "" : "s"}
                </span>
              </div>
            </div>
          );
        })}
      </div>

      {picked.length < 4 && manifest && (
        <div style={{ marginTop: 24 }}>
          <h2 style={{ fontSize: 16 }}>Add another pitch</h2>
          <div className="grid-cards">
            {manifest.pitches
              .filter((p) => !picked.includes(p.pitch_id))
              .map((p) => (
                <PitchCard
                  key={p.pitch_id}
                  pitch={p}
                  selectMode
                  onToggleSelect={togglePick}
                />
              ))}
          </div>
        </div>
      )}
    </div>
  );
}
