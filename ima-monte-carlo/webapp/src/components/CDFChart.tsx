import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { Distribution } from "@/data/types";
import { COLORS } from "@/lib/chartTheme";
import { fmtCurrency, fmtPercent } from "@/lib/format";

interface Props {
  distribution: Distribution;
  currentPrice: number | null;
}

export function CDFChart({ distribution, currentPrice }: Props) {
  const data = distribution.cdf_points.map((p) => ({
    x: p.value,
    cumulative: p.cumulative_prob,
  }));
  // Find P(FV > current) by interpolating at currentPrice
  let probAbove: number | null = null;
  if (currentPrice !== null && currentPrice !== undefined && data.length > 1) {
    for (let i = 0; i < data.length - 1; i++) {
      if (data[i].x <= currentPrice && data[i + 1].x >= currentPrice) {
        const t =
          (currentPrice - data[i].x) /
          ((data[i + 1].x - data[i].x) || 1);
        const below = data[i].cumulative + t * (data[i + 1].cumulative - data[i].cumulative);
        probAbove = 1 - below;
        break;
      }
    }
    if (probAbove === null) {
      probAbove = currentPrice <= (data[0]?.x ?? -Infinity) ? 1.0 : 0.0;
    }
  }

  return (
    <div className="card">
      <h2>Cumulative Probability</h2>
      <div style={{ fontSize: 13, color: "var(--text-muted)", marginBottom: 8 }}>
        {probAbove !== null
          ? `P(fair value > current) = ${fmtPercent(probAbove, 0)}`
          : "Probability of upside conditional on current price"}
      </div>
      <ResponsiveContainer width="100%" height={280}>
        <LineChart data={data} margin={{ top: 10, right: 10, left: 0, bottom: 4 }}>
          <CartesianGrid stroke="#EEE" />
          <XAxis
            dataKey="x"
            type="number"
            domain={["dataMin", "dataMax"]}
            tickFormatter={(v) => fmtCurrency(v as number)}
            fontSize={11}
            tickCount={8}
          />
          <YAxis
            domain={[0, 1]}
            tickFormatter={(v) => `${Math.round((v as number) * 100)}%`}
            fontSize={11}
            tickLine={false}
            axisLine={false}
          />
          <Tooltip
            formatter={(value: number) => [fmtPercent(value, 1), "Cumulative"]}
            labelFormatter={(label) => fmtCurrency(label as number)}
          />
          <Line
            type="monotone"
            dataKey="cumulative"
            stroke={COLORS.blue}
            strokeWidth={2}
            dot={false}
            isAnimationActive={false}
          />
          {[0.1, 0.25, 0.5, 0.75, 0.9].map((y) => (
            <ReferenceLine key={y} y={y} stroke={COLORS.grayBorder} strokeDasharray="2 4" />
          ))}
          {currentPrice !== null && currentPrice !== undefined && (
            <ReferenceLine
              x={currentPrice}
              stroke={COLORS.ink}
              strokeWidth={2}
              label={{
                value: `Current ${fmtCurrency(currentPrice)}`,
                position: "top",
                fontSize: 11,
              }}
            />
          )}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
