import type { Concentration } from "@/data/types";
import { fmtPercent } from "@/lib/format";

export function DriverConcentrationCard({ concentration }: { concentration?: Concentration }) {
  if (!concentration) return null;
  const focused = concentration.is_concentrated;
  const cls = focused ? "focused" : "diffuse";
  const top = concentration.top_drivers ?? [];
  return (
    <div className={`card concentration-card ${cls} section`}>
      <h2>{focused ? "Focused thesis" : "Diffuse drivers"}</h2>
      <p style={{ margin: "0 0 12px", lineHeight: 1.55 }}>{concentration.message}</p>
      {top.length > 0 && (
        <table className="compact-table">
          <thead>
            <tr>
              <th>Input</th>
              <th className="num">Variance share</th>
            </tr>
          </thead>
          <tbody>
            {top.map(([name, share]) => {
              const display = name.startsWith("catalyst:")
                ? `Catalyst · ${name.slice(9)}`
                : name;
              return (
                <tr key={name}>
                  <td>{display}</td>
                  <td className="num">{fmtPercent(share, 0)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
    </div>
  );
}
