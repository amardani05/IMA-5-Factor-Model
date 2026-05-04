import type { InputAnalysis } from "@/data/types";
import {
  Bar,
  BarChart,
  ReferenceLine,
  ResponsiveContainer,
  XAxis,
  YAxis,
} from "recharts";
import { COLORS } from "@/lib/chartTheme";

function MiniHistogram({ input }: { input: InputAnalysis }) {
  const data = input.histogram.edges.slice(0, -1).map((edge, i) => ({
    midpoint: (edge + input.histogram.edges[i + 1]) / 2,
    count: input.histogram.counts[i],
  }));
  const sm = input.sampled_summary;
  return (
    <div className="card" style={{ padding: 12 }}>
      <div style={{ fontSize: 12, fontWeight: 600 }}>{input.label}</div>
      <div style={{ fontSize: 11, color: "var(--text-muted)", marginBottom: 6 }}>
        μ={fmtNum(sm.mean)} • P10={fmtNum(sm.p10)} • P90={fmtNum(sm.p90)}
      </div>
      <ResponsiveContainer width="100%" height={120}>
        <BarChart data={data} margin={{ top: 4, right: 4, left: 0, bottom: 0 }}>
          <XAxis
            dataKey="midpoint"
            type="number"
            domain={["dataMin", "dataMax"]}
            tick={{ fontSize: 9 }}
            tickFormatter={(v) => fmtNum(v as number)}
          />
          <YAxis hide />
          <Bar dataKey="count" fill={COLORS.blue} fillOpacity={0.55} isAnimationActive={false} />
          {sm.median !== null && (
            <ReferenceLine x={sm.median} stroke={COLORS.ink} strokeWidth={1.5} />
          )}
          {sm.p10 !== null && (
            <ReferenceLine x={sm.p10} stroke={COLORS.gray} strokeDasharray="3 3" />
          )}
          {sm.p90 !== null && (
            <ReferenceLine x={sm.p90} stroke={COLORS.gray} strokeDasharray="3 3" />
          )}
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

function fmtNum(v: number | null | undefined) {
  if (v === null || v === undefined || !Number.isFinite(v)) return "—";
  const abs = Math.abs(v);
  if (abs >= 1e9) return `${(v / 1e9).toFixed(1)}B`;
  if (abs >= 1e6) return `${(v / 1e6).toFixed(1)}M`;
  if (abs >= 1e3) return `${(v / 1e3).toFixed(1)}K`;
  if (abs < 1 && abs > 0) return v.toFixed(3);
  return v.toFixed(2);
}

export function InputDistributionsPanel({ inputs }: { inputs: InputAnalysis[] }) {
  if (!inputs || inputs.length === 0) return null;
  return (
    <div className="card section">
      <h2>Sampled Input Distributions</h2>
      <div className="input-grid">
        {inputs.map((inp) => (
          <MiniHistogram key={inp.name} input={inp} />
        ))}
      </div>
    </div>
  );
}
