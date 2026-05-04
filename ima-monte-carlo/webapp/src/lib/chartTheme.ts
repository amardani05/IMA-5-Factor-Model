// Shared chart palette so every component reads the same colours.

export const COLORS = {
  green: "#2E7D32",
  greenSoft: "rgba(50, 180, 80, 0.18)",
  red: "#C62828",
  redSoft: "rgba(220, 50, 50, 0.18)",
  amber: "#F9A825",
  blue: "#1565C0",
  blueSoft: "rgba(21, 101, 192, 0.20)",
  gray: "#4A4A4A",
  grayLight: "#CCCCCC",
  grayBorder: "#E0E0E0",
  ink: "#1A1A1A",
  bg: "#FAFAFA",
  cardBg: "#FFFFFF",
  textMuted: "#666666",
};

export const SEVERITY_COLOR: Record<string, string> = {
  warning: COLORS.red,
  caution: COLORS.amber,
  info: COLORS.blue,
};

export const CHART_FONT = "Inter, system-ui, -apple-system, sans-serif";
