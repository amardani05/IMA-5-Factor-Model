import {
  Bar,
  BarChart,
  ReferenceLine,
  ResponsiveContainer,
  XAxis,
  YAxis,
} from "recharts";
import type { CalibrationEntry, HistoricalContextSummary } from "@/data/types";
import { COLORS, SEVERITY_COLOR } from "@/lib/chartTheme";
import { fmtMetric } from "@/lib/format";

function buildHistogram(values: (number | null)[] | undefined, bins = 20) {
  if (!values || values.length === 0) return [];
  const finite = values.filter((v): v is number => v !== null && Number.isFinite(v));
  if (finite.length === 0) return [];
  const lo = Math.min(...finite);
  const hi = Math.max(...finite);
  const span = hi - lo || 1;
  const buckets: { midpoint: number; count: number }[] = [];
  for (let i = 0; i < bins; i++) {
    const start = lo + (span * i) / bins;
    const end = lo + (span * (i + 1)) / bins;
    const count = finite.filter(
      (v) => v >= start && (i === bins - 1 ? v <= end : v < end),
    ).length;
    buckets.push({ midpoint: (start + end) / 2, count });
  }
  return buckets;
}

function MiniContext({
  ctx,
  unit,
  analyst,
  title,
}: {
  ctx: HistoricalContextSummary;
  unit: "ratio" | "multiple";
  analyst: { mean: number | null; p10: number | null; p90: number | null };
  title: string;
}) {
  const data = buildHistogram(ctx.raw_values);
  return (
    <div>
      <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 4 }}>{title}</div>
      <div style={{ fontSize: 11, color: "var(--text-muted)" }}>
        n={ctx.n_observations}
        {ctx.lookback_years ? ` • ${ctx.lookback_years.toFixed(1)}y lookback` : ""} •
        median {fmtMetric(ctx.median, unit)} • range {fmtMetric(ctx.min, unit)} →{" "}
        {fmtMetric(ctx.max, unit)}
      </div>
      <ResponsiveContainer width="100%" height={140}>
        <BarChart data={data} margin={{ top: 6, right: 4, left: 0, bottom: 0 }}>
          <XAxis
            dataKey="midpoint"
            type="number"
            domain={["dataMin", "dataMax"]}
            tickFormatter={(v) => fmtMetric(v as number, unit, 0)}
            fontSize={10}
          />
          <YAxis hide />
          <Bar dataKey="count" fill={COLORS.grayLight} fillOpacity={0.85} isAnimationActive={false} />
          {ctx.median !== null && (
            <ReferenceLine x={ctx.median} stroke={COLORS.ink} strokeWidth={1.2} />
          )}
          {analyst.mean !== null && (
            <ReferenceLine x={analyst.mean} stroke={COLORS.blue} strokeWidth={2} />
          )}
          {analyst.p90 !== null && (
            <ReferenceLine
              x={analyst.p90}
              stroke={COLORS.green}
              strokeDasharray="3 3"
              strokeWidth={1.5}
            />
          )}
          {analyst.p10 !== null && (
            <ReferenceLine
              x={analyst.p10}
              stroke={COLORS.red}
              strokeDasharray="3 3"
              strokeWidth={1.5}
            />
          )}
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

function unitOf(metric: string): "ratio" | "multiple" {
  return metric.includes("multiple") ? "multiple" : "ratio";
}

export function CalibrationCard({ entry }: { entry: CalibrationEntry }) {
  const unit = unitOf(entry.metric);
  const analyst = entry.analyst_distribution_summary;
  const showCo = !!entry.company_context;
  const showSec = !!entry.sector_context;

  return (
    <div className="calibration-card">
      <h3>
        {entry.input}{" "}
        <span style={{ color: "var(--text-muted)", fontWeight: 400 }}>
          — {entry.label}
        </span>
      </h3>
      <div className="two-col">
        {showCo && entry.company_context && (
          <MiniContext
            ctx={entry.company_context}
            unit={unit}
            analyst={analyst}
            title={`Company history (${entry.company_context.entity})`}
          />
        )}
        {showSec && entry.sector_context && (
          <MiniContext
            ctx={entry.sector_context}
            unit={unit}
            analyst={analyst}
            title={entry.sector_context.entity}
          />
        )}
      </div>

      <div className="pct-row">
        {showCo && (
          <div className="group">
            <div className="l">Company percentiles</div>
            <div>
              bear {fmtPctPlace(entry.analyst_p10_company_percentile)} • base{" "}
              {fmtPctPlace(entry.analyst_mean_company_percentile)} • bull{" "}
              {fmtPctPlace(entry.analyst_p90_company_percentile)}
            </div>
          </div>
        )}
        {showSec && (
          <div className="group">
            <div className="l">Sector percentiles</div>
            <div>
              bear {fmtPctPlace(entry.analyst_p10_sector_percentile)} • base{" "}
              {fmtPctPlace(entry.analyst_mean_sector_percentile)} • bull{" "}
              {fmtPctPlace(entry.analyst_p90_sector_percentile)}
            </div>
          </div>
        )}
      </div>

      <div style={{ fontSize: 13 }}>
        <strong>Analyst:</strong> bear {fmtMetric(analyst.p10, unit)} / base{" "}
        {fmtMetric(analyst.mean, unit)} / bull {fmtMetric(analyst.p90, unit)}
      </div>

      {entry.reversion_note && (
        <p
          style={{
            fontStyle: "italic",
            color: "var(--text-muted)",
            fontSize: 13,
            marginTop: 10,
          }}
        >
          {entry.reversion_note}
        </p>
      )}

      {entry.warnings.length > 0 ? (
        <ul className="warning-list">
          {entry.warnings.map((w, i) => (
            <li key={i} className={w.severity}>
              <span className="glyph" style={{ color: SEVERITY_COLOR[w.severity] }}>
                {w.severity === "warning" ? "⚠" : w.severity === "caution" ? "•" : "ℹ"}
              </span>
              <span className="body">
                {w.message}
                {w.suggestion && <span className="suggestion">→ {w.suggestion}</span>}
              </span>
            </li>
          ))}
        </ul>
      ) : (
        <div style={{ marginTop: 10 }}>
          <span className="chip success">✓ Within historical norms</span>
        </div>
      )}
    </div>
  );
}

function fmtPctPlace(v: number | null | undefined): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return "—";
  return `${Math.round(v)}th`;
}

export function CalibrationPanel({ entries }: { entries: CalibrationEntry[] }) {
  if (!entries || entries.length === 0) return null;
  // Sort: warnings first, cautions next, info/no-warning last
  const rank = (s: string) => (s === "warning" ? 0 : s === "caution" ? 1 : 2);
  const sorted = [...entries].sort((a, b) => rank(a.max_severity) - rank(b.max_severity));
  return (
    <div className="section">
      <h2>Historical Calibration</h2>
      {sorted.map((e) => (
        <CalibrationCard key={e.input} entry={e} />
      ))}
    </div>
  );
}
