// JSON shape exported by webapp_export.py. Keep in sync with that module.
// Optional fields appear when their underlying source data is present;
// always check before destructuring.

export interface Manifest {
  pitches: PitchSummary[];
  generated_at: string;
}

export interface PitchSummary {
  pitch_id: string;
  ticker: string;
  model_type: string;
  current_price: number | null;
  expected_return: number | null;
  probability_upside: number | null;
  risk_reward_ratio: number | null;
  generated_at: string;
  thesis_summary: string;
}

export interface PitchData {
  pitch_id: string;
  ticker: string;
  current_price: number | null;
  model_type: string;
  shares_outstanding: number | null;
  is_draft: boolean;
  generated_at: string;
  n_simulations: number;
  random_seed: number | null;
  thesis_statement: string;
  thesis_validation: ThesisValidation;
  distribution: Distribution;
  probability_analysis: ProbabilityAnalysis;
  inputs: InputAnalysis[];
  tornado?: TornadoEntry[];
  calibration?: CalibrationEntry[];
  concentration?: Concentration;
  catalysts?: Catalyst[];
  scenarios?: Record<string, ScenarioResult>;
}

export interface ThesisValidation {
  valid?: boolean;
  warnings?: string[];
  word_count?: number;
  driver_count_estimate?: number;
  is_placeholder?: boolean;
}

export interface Histogram {
  edges: number[];
  counts: number[];
}

export interface Distribution {
  mean: number | null;
  median: number | null;
  std: number | null;
  min: number | null;
  max: number | null;
  percentiles: Record<string, number | null>;
  histogram: Histogram;
  cdf_points: { value: number; cumulative_prob: number }[];
}

export interface ProbabilityAnalysis {
  expected_return: number | null;
  probability_upside: number | null;
  probability_above_20pct: number | null;
  probability_below_20pct: number | null;
  upside_capture: number | null;
  downside_capture: number | null;
  risk_reward_ratio: number | null;
  var_5: number | null;
  var_10: number | null;
  cvar_5: number | null;
  max_loss_pct: number | null;
  probability_above_cost?: number | null;
}

export interface InputAnalysis {
  name: string;
  label: string;
  type: string;
  spec: Record<string, unknown>;
  sampled_summary: {
    mean: number | null;
    median: number | null;
    std: number | null;
    p5: number | null;
    p10: number | null;
    p25: number | null;
    p75: number | null;
    p90: number | null;
    p95: number | null;
  };
  histogram: Histogram;
}

export interface TornadoEntry {
  input: string;
  label: string;
  p10_fair_value: number | null;
  p90_fair_value: number | null;
  input_p10: number | null;
  input_p90: number | null;
  variance_share: number;
}

export type Severity = "info" | "caution" | "warning";

export interface CalibrationWarning {
  severity: Severity;
  message: string;
  suggestion?: string | null;
}

export interface HistoricalContextSummary {
  entity: string;
  n_observations: number;
  lookback_years?: number | null;
  mean: number | null;
  median: number | null;
  std: number | null;
  min: number | null;
  max: number | null;
  p5: number | null;
  p10: number | null;
  p25: number | null;
  p75: number | null;
  p90: number | null;
  p95: number | null;
  long_run_mean?: number | null;
  recent_mean?: number | null;
  mean_reversion_implied?: number | null;
  raw_values?: (number | null)[];
}

export interface CalibrationEntry {
  input: string;
  metric: string;
  label: string;
  analyst_distribution_summary: {
    mean: number | null;
    p10: number | null;
    p90: number | null;
    median?: number | null;
    std?: number | null;
  };
  warnings: CalibrationWarning[];
  reversion_note: string | null;
  max_severity: Severity;
  company_context?: HistoricalContextSummary;
  sector_context?: HistoricalContextSummary;
  analyst_mean_company_percentile?: number | null;
  analyst_p90_company_percentile?: number | null;
  analyst_p10_company_percentile?: number | null;
  analyst_mean_sector_percentile?: number | null;
  analyst_p90_sector_percentile?: number | null;
  analyst_p10_sector_percentile?: number | null;
}

export interface Concentration {
  is_concentrated: boolean;
  top_drivers: [string, number][];
  concentration_pct: number;
  message: string;
}

export interface CatalystOutcome {
  probability: number | null;
  value_impact: number | null;
  impact_type: string;
}

export interface Catalyst {
  name: string;
  outcomes: Record<string, CatalystOutcome>;
}

export interface ScenarioResult {
  catalyst: string;
  outcome_label: string;
  mean_fair_value: number | null;
}
