import { Link } from "react-router-dom";

export function AboutPage() {
  return (
    <div className="container">
      <div className="page-header">
        <div>
          <h1>About IMA Monte Carlo</h1>
          <p>Methodology, principles, and limitations.</p>
        </div>
        <Link to="/" className="chip">
          ← Back to pitches
        </Link>
      </div>

      <div className="card section">
        <h2>What this is</h2>
        <p>
          The Illinois Investment Management Academy (IMA) is a student-run
          fundamental investment program. This dashboard publishes Monte Carlo
          analyses of IMA pitches: probability-weighted fair values, sensitivity
          decompositions, and calibration of analyst inputs against historical
          distributions of the same metrics.
        </p>
        <p>
          Each pitch on this site comes from a Python pipeline. Analysts declare
          distributional assumptions for each model input — revenue growth,
          margins, multiples, terminal growth, etc. — and the engine runs 50,000+
          simulations to produce a fair-value distribution.
        </p>
      </div>

      <div className="card section">
        <h2>How simulations work</h2>
        <p>
          Each iteration draws one value per input from its declared distribution
          (triangular, normal, uniform, lognormal, truncated, point, discrete,
          empirical). Correlated inputs are sampled via a Gaussian copula —
          marginals stay as declared, but the joint distribution respects the
          analyst-supplied correlation matrix. Each draw is fed through the
          valuation model (multiples, DCF, SOTP, or custom) to produce a
          fair-value-per-share. We aggregate across all draws to get the output
          distribution and derived statistics: probability of upside, VaR, CVaR,
          risk/reward, and the like.
        </p>
        <p>
          Two sensitivity views are computed. The tornado view freezes every
          other input at its sampled median and varies one input from its P10 to
          P90, recording the resulting fair value swing. The variance
          contribution view uses the squared correlation between each sampled
          input and the resulting fair value as a Sobol-like proxy for the share
          of output variance the input explains.
        </p>
      </div>

      <div className="card section">
        <h2>Why historical calibration matters</h2>
        <p>
          For each input that maps to a historical metric (revenue growth, EBITDA
          margin, ROIC, multiples), we pull the company's own quarterly history
          and a sector-peer pool from yfinance. The dashboard then shows where
          the analyst's distribution sits inside both — a 90th-percentile bull
          case is flagged differently from a 30th-percentile one.
        </p>
        <p>
          This is descriptive, not prescriptive. The analyst is welcome to
          project anything. Calibration just makes it explicit when a projection
          requires structural-break justification.
        </p>
      </div>

      <div className="card section">
        <h2>Why driver concentration is checked</h2>
        <p>
          A pitch where two inputs explain &gt;80% of output variance is
          effectively a two-driver bet. That's not a defect — it usually means
          the thesis is focused, which is good — but it tells the analyst (and
          the audience) which inputs deserve scrutiny.
        </p>
        <p>
          The opposite case — diffuse drivers — usually means either the thesis
          is multi-faceted (and should say so) or some inputs are over-specified
          as distributions when they could be point estimates without changing
          the conclusion.
        </p>
      </div>

      <div className="card section">
        <h2>Limitations</h2>
        <ul style={{ lineHeight: 1.7 }}>
          <li>
            <strong>Inputs are subjective.</strong> Monte Carlo doesn't make
            opinions defensible, it makes them legible. Garbage in, garbage out
            still applies.
          </li>
          <li>
            <strong>Historical data has gaps for small caps.</strong> yfinance
            quarterly fundamentals can be sparse or missing for smaller IJR
            constituents. The dashboard surfaces "n=X" alongside every
            distribution so you can judge weight.
          </li>
          <li>
            <strong>Mean reversion isn't universal.</strong> Margins, ROIC, and
            multiples mean-revert empirically. But persistent moats and
            structural breaks can defeat reversion. Calibration warnings are a
            prompt to defend, not a veto.
          </li>
          <li>
            <strong>Outputs are decision support, not predictions.</strong> A
            65% probability of upside is not a 65% chance of being right — it's
            a 65% chance, conditional on the analyst's input distributions
            being correct.
          </li>
        </ul>
      </div>

      <div className="card section">
        <h2>Source</h2>
        <p>
          The pipeline (Python) and dashboard (React + Vite) are open-source.
          See the <a href="https://github.com" target="_blank" rel="noreferrer">GitHub repo</a>
          for the full source, including the calibration registry, peer
          aggregation logic, and dashboard components.
        </p>
      </div>
    </div>
  );
}
