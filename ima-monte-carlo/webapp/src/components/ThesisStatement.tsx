import type { PitchData } from "@/data/types";

export function ThesisStatement({ data }: { data: PitchData }) {
  const { thesis_statement, thesis_validation } = data;
  const wc = thesis_validation?.word_count ?? 0;
  const drivers = thesis_validation?.driver_count_estimate ?? 0;
  const warnings = thesis_validation?.warnings ?? [];
  const isPlaceholder = thesis_validation?.is_placeholder;

  return (
    <div className="card section">
      <h2>Thesis</h2>
      {isPlaceholder ? (
        <p style={{ color: "var(--text-muted)", fontStyle: "italic" }}>
          No thesis statement provided.
        </p>
      ) : (
        <blockquote className="thesis-quote">{thesis_statement}</blockquote>
      )}
      <div className="thesis-meta">
        <span className="chip muted">{wc} words</span>
        <span className="chip muted">~{drivers} driver{drivers === 1 ? "" : "s"}</span>
        {warnings.length === 0 ? (
          <span className="chip success">✓ Within target</span>
        ) : (
          warnings.map((w, i) => (
            <span key={i} className="chip caution">
              ⚠ {w}
            </span>
          ))
        )}
      </div>
    </div>
  );
}
