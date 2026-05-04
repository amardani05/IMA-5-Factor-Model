import type { PitchData } from "@/data/types";
import { fmtCurrency, fmtPercent, fmtMultiplier } from "@/lib/format";

export function KeyMetricsHeader({ data }: { data: PitchData }) {
  const pa = data.probability_analysis;
  const dist = data.distribution;
  const cp = data.current_price ?? 0;
  const meanRet = pa.expected_return;
  const meanFv = dist.mean;
  const upsideClass =
    meanRet === null || meanRet === undefined
      ? ""
      : meanRet >= 0
      ? "return-positive"
      : "return-negative";

  return (
    <div className="metrics-grid">
      <div className="metric-card">
        <div className="label">Mean Fair Value</div>
        <div className={`value ${upsideClass}`}>{fmtCurrency(meanFv)}</div>
        <div className="sub">
          {fmtPercent(meanRet, 1, true)} vs current ({fmtCurrency(cp)})
        </div>
      </div>
      <div className="metric-card">
        <div className="label">Probability of Upside</div>
        <div className="value">{fmtPercent(pa.probability_upside, 0)}</div>
        <div className="sub">P(fair value &gt; current price)</div>
      </div>
      <div className="metric-card">
        <div className="label">Risk / Reward</div>
        <div className="value">{fmtMultiplier(pa.risk_reward_ratio, 2)}</div>
        <div className="sub">avg upside / avg downside</div>
      </div>
      <div className="metric-card">
        <div className="label">5% VaR</div>
        <div className="value">{fmtCurrency(pa.var_5)}</div>
        <div className="sub">5th-percentile fair value</div>
      </div>
    </div>
  );
}
