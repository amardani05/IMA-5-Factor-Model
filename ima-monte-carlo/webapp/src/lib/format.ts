// Display helpers used across the dashboard. All accept null/undefined.

export function fmtCurrency(value: number | null | undefined, fractionDigits = 2): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  const abs = Math.abs(value);
  if (abs >= 1e9) return `$${(value / 1e9).toFixed(1)}B`;
  if (abs >= 1e6) return `$${(value / 1e6).toFixed(1)}M`;
  if (abs >= 1e3) return `$${(value / 1e3).toFixed(1)}K`;
  return `$${value.toFixed(fractionDigits)}`;
}

export function fmtPrice(value: number | null | undefined, fractionDigits = 2): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  return `$${value.toFixed(fractionDigits)}`;
}

export function fmtPercent(
  value: number | null | undefined,
  fractionDigits = 1,
  withSign = false,
): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  const sign = withSign && value > 0 ? "+" : "";
  return `${sign}${(value * 100).toFixed(fractionDigits)}%`;
}

export function fmtMultiplier(value: number | null | undefined, fractionDigits = 2): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  return `${value.toFixed(fractionDigits)}x`;
}

export function fmtMetric(
  value: number | null | undefined,
  unit: "ratio" | "multiple",
  fractionDigits = 1,
): string {
  if (unit === "ratio") return fmtPercent(value, fractionDigits);
  return fmtMultiplier(value, fractionDigits);
}

export function fmtRelativeTime(iso: string | null | undefined): string {
  if (!iso) return "";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "";
  const seconds = (Date.now() - d.getTime()) / 1000;
  if (seconds < 60) return "just now";
  const minutes = seconds / 60;
  if (minutes < 60) return `${Math.round(minutes)} min ago`;
  const hours = minutes / 60;
  if (hours < 24) return `${Math.round(hours)}h ago`;
  const days = hours / 24;
  if (days < 7) return `${Math.round(days)}d ago`;
  return d.toLocaleDateString();
}

export function fmtAbsDate(iso: string | null | undefined): string {
  if (!iso) return "";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "";
  return d.toLocaleString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function modelTypeLabel(t: string): string {
  switch ((t || "").toLowerCase()) {
    case "dcf": return "DCF";
    case "multiples": return "Multiples";
    case "sotp": return "SOTP";
    case "custom": return "Custom";
    default: return t || "Unknown";
  }
}

export function clamp(v: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, v));
}

export function returnCategory(expected: number | null | undefined): "high" | "mid" | "negative" {
  if (expected === null || expected === undefined || !Number.isFinite(expected)) return "mid";
  if (expected < 0) return "negative";
  if (expected > 0.30) return "high";
  return "mid";
}
