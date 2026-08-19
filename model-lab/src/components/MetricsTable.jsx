import { Check } from "lucide-react";
import { MODEL_COLORS, bestModelsForMetric, formatMetric } from "../utils/models";

const METRIC_ROWS = [
  { key: "recall", label: "Recall" },
  { key: "precision", label: "Precision" },
  { key: "f1", label: "F1" },
  { key: "roc_auc", label: "ROC-AUC" },
  { key: "pr_auc", label: "PR-AUC" },
];

export default function MetricsTable({ metrics, modelOrder, modelLabels }) {
  return (
    <div style={{ overflowX: "auto" }}>
      <table style={{ fontSize: 13.5 }}>
        <thead>
          <tr>
            <th style={{ textAlign: "left", padding: "8px 14px 8px 0", color: "var(--color-text-muted)", fontWeight: 500 }}>
              Metric
            </th>
            {modelOrder.map((m) => (
              <th
                key={m}
                style={{
                  textAlign: "right",
                  padding: "8px 14px",
                  fontWeight: 600,
                  color: MODEL_COLORS[m],
                  whiteSpace: "nowrap",
                }}
              >
                <span
                  style={{
                    display: "inline-block",
                    width: 8,
                    height: 8,
                    borderRadius: "50%",
                    background: MODEL_COLORS[m],
                    marginRight: 6,
                  }}
                />
                {modelLabels[m]}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {METRIC_ROWS.map(({ key, label }) => {
            const row = Object.fromEntries(modelOrder.map((m) => [m, metrics[m][key]]));
            const best = bestModelsForMetric(row, modelOrder);
            return (
              <tr key={key} style={{ borderTop: "1px solid var(--color-border)" }}>
                <td style={{ padding: "9px 14px 9px 0", color: "var(--color-text-muted)" }}>{label}</td>
                {modelOrder.map((m) => {
                  const isBest = best.has(m);
                  return (
                    <td
                      key={m}
                      className="mono"
                      style={{
                        textAlign: "right",
                        padding: "9px 14px",
                        fontWeight: isBest ? 700 : 400,
                        background: isBest ? "var(--color-low-tint)" : "transparent",
                        borderRadius: 6,
                      }}
                    >
                      {isBest && <Check size={12} style={{ verticalAlign: -1, marginRight: 4, color: "var(--color-low)" }} />}
                      {formatMetric(row[m])}
                    </td>
                  );
                })}
              </tr>
            );
          })}
        </tbody>
      </table>
      <p style={{ fontSize: 11.5, color: "var(--color-text-muted)", marginTop: 12 }}>
        Best value per row is marked -- no model wins on every metric, so there is no single "best model" here.
      </p>
    </div>
  );
}
