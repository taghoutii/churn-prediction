// Shared risk-tier logic. Kept in one place so the signal-bar indicator,
// the overview chart, and the customer list all agree on the same bands.

export const RISK_COLORS = {
  low: "#16A34A",
  medium: "#F59E0B",
  high: "#E30613",
};

export const RISK_TINTS = {
  low: "#EAFAF0",
  medium: "#FEF6E7",
  high: "#FDECEB",
};

export const RISK_LABELS = {
  low: "Low risk",
  medium: "Medium risk",
  high: "High risk",
};

export function riskTierFromBounds(probability, bounds) {
  if (probability < bounds.low_max) return "low";
  if (probability < bounds.medium_max) return "medium";
  return "high";
}

export function tierToBars(tier) {
  return tier === "high" ? 3 : tier === "medium" ? 2 : 1;
}

export function formatPercent(probability, digits = 1) {
  return `${(probability * 100).toFixed(digits)}%`;
}

// The decision (flagged / not flagged) is intentionally NOT precomputed in
// the exported data -- it's derived here, live, from the raw probability
// and the working threshold, so the UI can make clear that a probability
// and a business decision are two different things.
export function isFlagged(probability, decisionThreshold) {
  return probability >= decisionThreshold;
}
