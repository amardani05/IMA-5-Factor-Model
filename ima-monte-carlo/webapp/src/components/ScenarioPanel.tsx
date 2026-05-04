import type { Catalyst, ScenarioResult } from "@/data/types";
import { fmtCurrency, fmtPercent } from "@/lib/format";

interface Props {
  catalysts: Catalyst[];
  scenarios?: Record<string, ScenarioResult>;
  meanFairValue: number | null;
}

export function ScenarioPanel({ catalysts, scenarios, meanFairValue }: Props) {
  if (!catalysts || catalysts.length === 0) return null;
  return (
    <div className="card section">
      <h2>Catalysts</h2>
      <div style={{ fontSize: 12, color: "var(--text-muted)", marginBottom: 12 }}>
        Probability-weighted outcomes layered on top of the base distribution.
      </div>
      {catalysts.map((cat) => (
        <div key={cat.name} style={{ marginBottom: 18 }}>
          <h3 style={{ fontSize: 14, margin: "0 0 8px" }}>{cat.name}</h3>
          <table className="compact-table">
            <thead>
              <tr>
                <th>Outcome</th>
                <th className="num">Probability</th>
                <th className="num">Impact</th>
                <th className="num">Mean FV | outcome</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(cat.outcomes).map(([label, oc]) => {
                const key = `${cat.name} | ${label}`;
                const sc = scenarios?.[key];
                return (
                  <tr key={label}>
                    <td>{label}</td>
                    <td className="num">{fmtPercent(oc.probability, 0)}</td>
                    <td className="num">
                      {oc.impact_type === "multiplicative"
                        ? `${(oc.value_impact ?? 0).toFixed(2)}x`
                        : fmtCurrency(oc.value_impact)}
                    </td>
                    <td className="num">{fmtCurrency(sc?.mean_fair_value)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      ))}
      {meanFairValue !== null && (
        <div style={{ fontSize: 12, color: "var(--text-muted)", marginTop: 4 }}>
          Overall mean fair value (post-catalyst): {fmtCurrency(meanFairValue)}
        </div>
      )}
    </div>
  );
}
