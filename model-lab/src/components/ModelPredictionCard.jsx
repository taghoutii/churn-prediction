import { ArrowDownRight, ArrowUpRight } from "lucide-react";
import { MODEL_COLORS, formatPercent } from "../utils/models";

export default function ModelPredictionCard({ modelKey, modelLabel, prediction }) {
  const { predicted_proba, predicted_class, threshold, top_factors } = prediction;
  const flagged = predicted_class === 1;

  return (
    <div
      style={{
        background: "var(--color-surface)",
        border: `1px solid var(--color-border)`,
        borderTop: `3px solid ${MODEL_COLORS[modelKey]}`,
        borderRadius: "var(--radius-lg)",
        boxShadow: "var(--shadow-card)",
        padding: 18,
        display: "flex",
        flexDirection: "column",
        gap: 12,
        minWidth: 0,
      }}
    >
      <div>
        <h4 style={{ fontSize: 14, color: MODEL_COLORS[modelKey] }}>{modelLabel}</h4>
      </div>

      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
        <div>
          <div style={{ fontSize: 11, color: "var(--color-text-muted)" }}>Predicted probability</div>
          <div className="mono" style={{ fontSize: 22, fontWeight: 700 }}>{formatPercent(predicted_proba)}</div>
        </div>
        <span
          style={{
            fontSize: 11.5,
            fontWeight: 600,
            padding: "4px 9px",
            borderRadius: 999,
            color: flagged ? "var(--color-high)" : "var(--color-low)",
            background: flagged ? "var(--color-high-tint)" : "var(--color-low-tint)",
            whiteSpace: "nowrap",
          }}
        >
          {flagged ? "Churn" : "No churn"}
        </span>
      </div>

      <div style={{ fontSize: 11, color: "var(--color-text-muted)" }}>
        own threshold <span className="mono">{formatPercent(threshold, 0)}</span>
      </div>

      <div style={{ borderTop: "1px solid var(--color-border)", paddingTop: 10, display: "flex", flexDirection: "column", gap: 7 }}>
        {top_factors.slice(0, 6).map((f, i) => {
          const up = f.direction === "increases_risk";
          const Icon = up ? ArrowUpRight : ArrowDownRight;
          return (
            <div key={i} style={{ display: "flex", alignItems: "flex-start", gap: 6, fontSize: 12 }}>
              <Icon size={13} style={{ marginTop: 1, flexShrink: 0, color: up ? "var(--color-high)" : "var(--color-low)" }} />
              <span style={{ color: "var(--color-text)" }}>{f.short_label}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
