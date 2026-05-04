import type { TornadoEntry } from "@/data/types";
import { COLORS } from "@/lib/chartTheme";
import { fmtPercent } from "@/lib/format";

interface Props {
  entries: TornadoEntry[];
  minShare?: number; // entries below this fold into "Other"
}

const PALETTE = [
  COLORS.blue,
  COLORS.green,
  COLORS.amber,
  "#6A1B9A",
  "#00838F",
  "#5D4037",
  "#455A64",
  "#AD1457",
];

export function VarianceContributionChart({ entries, minShare = 0.03 }: Props) {
  if (!entries || entries.length === 0) return null;

  const total = entries.reduce((s, e) => s + Math.max(0, e.variance_share), 0);
  if (total <= 0) return null;

  const sorted = [...entries].sort((a, b) => b.variance_share - a.variance_share);
  const items: { label: string; share: number; color: string }[] = [];
  let other = 0;
  for (const e of sorted) {
    const share = e.variance_share / total;
    if (share < minShare) {
      other += share;
    } else {
      items.push({ label: e.label, share, color: PALETTE[items.length % PALETTE.length] });
    }
  }
  if (other > 0) items.push({ label: "Other", share: other, color: COLORS.gray });

  return (
    <div className="card">
      <h2>Variance Contribution</h2>
      <div style={{ fontSize: 12, color: "var(--text-muted)", marginBottom: 12 }}>
        Each input's share of total output variance. Inputs below 3% combined as "Other".
      </div>
      <div
        style={{
          display: "flex",
          height: 28,
          width: "100%",
          borderRadius: 6,
          overflow: "hidden",
          border: "1px solid var(--gray-border)",
        }}
      >
        {items.map((it) => (
          <div
            key={it.label}
            title={`${it.label}: ${fmtPercent(it.share, 0)}`}
            style={{
              width: `${it.share * 100}%`,
              background: it.color,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              color: "white",
              fontSize: 11,
              fontWeight: 600,
            }}
          >
            {it.share >= 0.07 ? fmtPercent(it.share, 0) : ""}
          </div>
        ))}
      </div>
      <table className="compact-table" style={{ marginTop: 14 }}>
        <thead>
          <tr>
            <th>Input</th>
            <th className="num">Share</th>
          </tr>
        </thead>
        <tbody>
          {items.map((it) => (
            <tr key={it.label}>
              <td>
                <span
                  style={{
                    display: "inline-block",
                    width: 10,
                    height: 10,
                    background: it.color,
                    borderRadius: 2,
                    marginRight: 8,
                  }}
                />
                {it.label}
              </td>
              <td className="num">{fmtPercent(it.share, 1)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
