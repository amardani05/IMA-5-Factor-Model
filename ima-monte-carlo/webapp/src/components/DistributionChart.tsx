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
import type { Distribution } from "@/data/types";
import { COLORS } from "@/lib/chartTheme";
import { fmtCurrency, fmtPercent } from "@/lib/format";

interface Props {
  distribution: Distribution;
  currentPrice: number | null;
  expectedReturn: number | null;
  probabilityUpside: number | null;
}

export function DistributionChart({
  distribution,
  currentPrice,
  expectedReturn,
  probabilityUpside,
}: Props) {
  const { histogram, percentiles } = distribution;
  const data = histogram.edges.slice(0, -1).map((edge, i) => ({
    binStart: edge,
    binEnd: histogram.edges[i + 1],
    midpoint: (edge + histogram.edges[i + 1]) / 2,
    count: histogram.counts[i],
    color:
      currentPrice !== null && currentPrice !== undefined && (edge + histogram.edges[i + 1]) / 2 >= currentPrice
        ? COLORS.green
        : COLORS.red,
  }));
  const p10 = percentiles["10"];
  const p50 = percentiles["50"] ?? distribution.median;
  const p90 = percentiles["90"];

  return (
    <div className="card">
      <h2>Fair Value Distribution</h2>
      <div style={{ fontSize: 13, color: "var(--text-muted)", marginBottom: 8 }}>
        P(undervalued) = {fmtPercent(probabilityUpside, 0)} • Expected return ={" "}
        {fmtPercent(expectedReturn, 1, true)}
      </div>
      <ResponsiveContainer width="100%" height={280}>
        <BarChart data={data} margin={{ top: 10, right: 10, left: 0, bottom: 4 }}>
          <CartesianGrid stroke="#EEE" />
          <XAxis
            dataKey="midpoint"
            type="number"
            domain={[
              data[0]?.binStart ?? 0,
              data[data.length - 1]?.binEnd ?? 1,
            ]}
            tickFormatter={(v) => fmtCurrency(v as number)}
            fontSize={11}
            tickCount={8}
          />
          <YAxis fontSize={11} tickLine={false} axisLine={false} />
          <Tooltip
            cursor={{ fill: "rgba(0,0,0,0.04)" }}
            formatter={(value: number) => [`${value} draws`, "Count"]}
            labelFormatter={(_, items) => {
              const item = items?.[0]?.payload as
                | { binStart: number; binEnd: number }
                | undefined;
              if (!item) return "";
              return `${fmtCurrency(item.binStart)} – ${fmtCurrency(item.binEnd)}`;
            }}
          />
          <Bar
            dataKey="count"
            isAnimationActive={false}
            shape={(props: any) => {
              const { x, y, width, height, payload } = props;
              return (
                <rect
                  x={x}
                  y={y}
                  width={width}
                  height={height}
                  fill={payload.color}
                  fillOpacity={0.55}
                />
              );
            }}
          />
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
          {p10 !== null && p10 !== undefined && (
            <ReferenceLine
              x={p10}
              stroke={COLORS.gray}
              strokeDasharray="3 3"
              label={{ value: `P10`, position: "top", fontSize: 10, fill: COLORS.gray }}
            />
          )}
          {p50 !== null && p50 !== undefined && (
            <ReferenceLine
              x={p50}
              stroke={COLORS.blue}
              strokeDasharray="3 3"
              label={{ value: `Median`, position: "top", fontSize: 10, fill: COLORS.blue }}
            />
          )}
          {p90 !== null && p90 !== undefined && (
            <ReferenceLine
              x={p90}
              stroke={COLORS.gray}
              strokeDasharray="3 3"
              label={{ value: `P90`, position: "top", fontSize: 10, fill: COLORS.gray }}
            />
          )}
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
