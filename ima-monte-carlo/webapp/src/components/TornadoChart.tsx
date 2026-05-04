import {
  Bar,
  BarChart,
  CartesianGrid,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { TornadoEntry } from "@/data/types";
import { COLORS } from "@/lib/chartTheme";
import { fmtCurrency } from "@/lib/format";

interface Props {
  entries: TornadoEntry[];
  centerValue: number | null;
}

export function TornadoChart({ entries, centerValue }: Props) {
  if (!entries || entries.length === 0) return null;
  const usable = entries.filter(
    (e) =>
      e.p10_fair_value !== null &&
      e.p90_fair_value !== null &&
      Number.isFinite(e.p10_fair_value as number) &&
      Number.isFinite(e.p90_fair_value as number),
  );
  if (usable.length === 0) {
    return (
      <div className="card">
        <h2>Tornado Sensitivity</h2>
        <p style={{ color: "var(--text-muted)" }}>
          No tornado data available (all inputs were point estimates).
        </p>
      </div>
    );
  }

  const data = usable.map((e) => {
    const lo = Math.min(e.p10_fair_value as number, e.p90_fair_value as number);
    const hi = Math.max(e.p10_fair_value as number, e.p90_fair_value as number);
    const center = centerValue ?? (lo + hi) / 2;
    return {
      label: e.label,
      input: e.input,
      negative: -(Math.max(0, center - lo)),
      positive: Math.max(0, hi - center),
      span: hi - lo,
      lo,
      hi,
      input_p10: e.input_p10,
      input_p90: e.input_p90,
    };
  });
  data.sort((a, b) => b.span - a.span);

  const height = Math.max(220, data.length * 28 + 40);

  return (
    <div className="card">
      <h2>Tornado Sensitivity</h2>
      <div style={{ fontSize: 12, color: "var(--text-muted)", marginBottom: 8 }}>
        Bars show fair value when each input swings P10 → P90, others held at median.
      </div>
      <ResponsiveContainer width="100%" height={height}>
        <BarChart
          data={data}
          layout="vertical"
          stackOffset="sign"
          margin={{ top: 10, right: 30, left: 80, bottom: 4 }}
        >
          <CartesianGrid stroke="#EEE" />
          <XAxis
            type="number"
            tickFormatter={(v) => fmtCurrency(((centerValue ?? 0) + (v as number)))}
            fontSize={11}
          />
          <YAxis
            dataKey="label"
            type="category"
            width={140}
            fontSize={11}
            tickLine={false}
            axisLine={false}
          />
          <Tooltip
            formatter={(value: number, name) => {
              const offset = centerValue ?? 0;
              return [fmtCurrency(offset + value), name === "negative" ? "Low" : "High"];
            }}
          />
          <ReferenceLine x={0} stroke={COLORS.ink} strokeWidth={1.2} />
          <Bar dataKey="negative" fill={COLORS.red} fillOpacity={0.75} stackId="a" isAnimationActive={false} />
          <Bar dataKey="positive" fill={COLORS.green} fillOpacity={0.75} stackId="a" isAnimationActive={false} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
