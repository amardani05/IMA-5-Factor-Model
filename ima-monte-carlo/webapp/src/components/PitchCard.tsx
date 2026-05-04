import { Link } from "react-router-dom";
import type { PitchSummary } from "@/data/types";
import {
  fmtPercent,
  fmtPrice,
  fmtRelativeTime,
  modelTypeLabel,
  returnCategory,
  clamp,
} from "@/lib/format";

interface Props {
  pitch: PitchSummary;
  selected?: boolean;
  onToggleSelect?: (id: string) => void;
  selectMode?: boolean;
  isDraft?: boolean;
}

export function PitchCard({ pitch, selected, onToggleSelect, selectMode, isDraft }: Props) {
  const cat = returnCategory(pitch.expected_return);
  const upside = pitch.probability_upside ?? 0;
  const card = (
    <div className={`pitch-card${selected ? " selected" : ""}`}>
      <div style={{ display: "flex", justifyContent: "space-between" }}>
        <span className="ticker">{pitch.ticker}</span>
        <div className="badges">
          <span className="chip muted">{modelTypeLabel(pitch.model_type)}</span>
          {isDraft && <span className="draft-badge">draft</span>}
        </div>
      </div>
      <div className="stat-row">
        <div className="stat">
          <span className="l">Expected return</span>
          <span
            className={`v ${cat === "negative" ? "return-negative" : "return-positive"}`}
          >
            {fmtPercent(pitch.expected_return, 1, true)}
          </span>
        </div>
        <div className="stat">
          <span className="l">Risk / Reward</span>
          <span className="v">
            {pitch.risk_reward_ratio !== null && Number.isFinite(pitch.risk_reward_ratio)
              ? `${pitch.risk_reward_ratio.toFixed(2)}x`
              : "—"}
          </span>
        </div>
      </div>
      <div>
        <span className="l" style={{ fontSize: 11, color: "var(--text-muted)" }}>
          P(upside)
        </span>
        <div className="metric-bar">
          <span style={{ width: `${clamp(upside * 100, 0, 100)}%` }} />
        </div>
        <span style={{ fontSize: 12, color: "var(--text-muted)" }}>
          {fmtPercent(upside, 0)} chance fair value &gt; current price ({fmtPrice(pitch.current_price)})
        </span>
      </div>
      {pitch.thesis_summary && (
        <p className="thesis-snip">{pitch.thesis_summary}</p>
      )}
      <footer>
        <span>{fmtRelativeTime(pitch.generated_at)}</span>
        {selectMode && (
          <button
            className="chip"
            onClick={(e) => {
              e.preventDefault();
              onToggleSelect?.(pitch.pitch_id);
            }}
          >
            {selected ? "Selected" : "+ Compare"}
          </button>
        )}
      </footer>
    </div>
  );

  if (selectMode) {
    return (
      <div onClick={() => onToggleSelect?.(pitch.pitch_id)} role="button" tabIndex={0}>
        {card}
      </div>
    );
  }

  return (
    <Link to={`/pitch/${pitch.pitch_id}`} style={{ textDecoration: "none", color: "inherit" }}>
      {card}
    </Link>
  );
}
