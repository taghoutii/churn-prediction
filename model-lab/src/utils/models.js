// Shared model metadata. Colors are defined here (not in the exported JSON)
// since they're a presentation concern -- one consistent color per model,
// reused across the metrics table, curves, confusion matrices, and
// prediction cards so a model is visually identifiable everywhere at once.
export const MODEL_COLORS = {
  logistic_regression: "#6366f1",
  xgboost: "#d97706",
  lightgbm: "#16a34a",
  voting_ensemble: "#db2777",
};

export function formatPercent(value, digits = 1) {
  return `${(value * 100).toFixed(digits)}%`;
}

export function formatMetric(value, digits = 3) {
  return value.toFixed(digits);
}

// For a metric table row like {logistic_regression: 0.91, xgboost: 0.77, ...},
// returns the set of model keys tied for the best (highest) value -- used to
// visually mark the best-per-metric without declaring one model "the winner".
export function bestModelsForMetric(row, modelOrder) {
  const values = modelOrder.map((m) => row[m]);
  const best = Math.max(...values);
  return new Set(modelOrder.filter((m) => Math.abs(row[m] - best) < 1e-9));
}
