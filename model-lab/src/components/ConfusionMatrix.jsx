import { MODEL_COLORS, formatPercent } from "../utils/models";

function Cell({ label, value, tone }) {
  const tones = {
    correct: { bg: "var(--color-low-tint)", fg: "var(--color-low)" },
    wrong: { bg: "var(--color-high-tint)", fg: "var(--color-high)" },
  };
  const t = tones[tone];
  return (
    <div
      style={{
        background: t.bg,
        borderRadius: "var(--radius-sm)",
        padding: "10px 8px",
        textAlign: "center",
      }}
    >
      <div className="mono" style={{ fontSize: 18, fontWeight: 700, color: t.fg }}>
        {value.toLocaleString()}
      </div>
      <div style={{ fontSize: 10.5, color: "var(--color-text-muted)", marginTop: 2 }}>{label}</div>
    </div>
  );
}

export default function ConfusionMatrix({ modelKey, modelLabel, matrix }) {
  const { tn, fp, fn, tp, threshold } = matrix;
  return (
    <div style={{ background: "var(--color-surface)", border: "1px solid var(--color-border)", borderRadius: "var(--radius-lg)", boxShadow: "var(--shadow-card)", padding: 18 }}>
      <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", marginBottom: 10 }}>
        <h4 style={{ fontSize: 13.5, color: MODEL_COLORS[modelKey] }}>{modelLabel}</h4>
        <span className="mono" style={{ fontSize: 11, color: "var(--color-text-muted)" }}>
          threshold {formatPercent(threshold, 0)}
        </span>
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 6 }}>
        <Cell label="True negative" value={tn} tone="correct" />
        <Cell label="False positive" value={fp} tone="wrong" />
        <Cell label="False negative" value={fn} tone="wrong" />
        <Cell label="True positive" value={tp} tone="correct" />
      </div>
    </div>
  );
}
