// "Balanced" always tracks the model's actual deployed decision threshold
// (meta.decision_threshold) rather than a hardcoded number, so it can't go
// stale the next time the model is retuned. Aggressive/Conservative are
// fixed reference points at the model's own 10:1 / 2:1 cost-ratio
// thresholds (see threshold/calibration analysis) -- refresh these if the
// model is retuned again.
function buildPresets(decisionThreshold) {
  return [
    { label: "Aggressive", value: 0.10 },
    { label: "Balanced", value: decisionThreshold },
    { label: "Conservative", value: 0.50 },
  ];
}

export default function ThresholdSlider({ value, onChange, decisionThreshold }) {
  const PRESETS = buildPresets(decisionThreshold);
  return (
    <div
      style={{
        background: "var(--color-surface)",
        border: "1px solid var(--color-border)",
        borderRadius: "var(--radius-lg)",
        boxShadow: "var(--shadow-card)",
        padding: 22,
      }}
    >
      <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between" }}>
        <h3 style={{ fontSize: 14.5 }}>Retention threshold</h3>
        <span
          data-testid="threshold-value"
          className="mono"
          style={{ fontSize: 22, fontWeight: 600, color: "var(--color-brand)" }}
        >
          {(value * 100).toFixed(0)}%
        </span>
      </div>
      <p style={{ fontSize: 13, color: "var(--color-text-muted)", margin: "4px 0 18px" }}>
        Customers whose estimated risk is at or above this level are surfaced for retention outreach below.
      </p>

      <input
        type="range"
        min={0.05}
        max={0.95}
        step={0.01}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        aria-label="Retention threshold"
        style={{ width: "100%", accentColor: "var(--color-brand)" }}
      />

      <div style={{ display: "flex", gap: 8, marginTop: 16 }}>
        {PRESETS.map((p) => {
          const active = Math.abs(value - p.value) < 0.005;
          return (
            <button
              key={p.label}
              onClick={() => onChange(p.value)}
              style={{
                flex: 1,
                padding: "9px 10px",
                borderRadius: "var(--radius-md)",
                border: active ? "1px solid var(--color-brand)" : "1px solid var(--color-border)",
                background: active ? "var(--color-brand-tint)" : "var(--color-surface)",
                color: active ? "var(--color-brand)" : "var(--color-text)",
                fontWeight: 600,
                fontSize: 13,
                cursor: "pointer",
              }}
            >
              {p.label}
              <div className="mono" style={{ fontWeight: 400, fontSize: 11.5, marginTop: 2 }}>
                {(p.value * 100).toFixed(0)}%
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}
