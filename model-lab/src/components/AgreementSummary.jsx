const ROWS = [
  { key: "all_4_agree", label: "All 4 agree", tone: "var(--color-low)", tint: "var(--color-low-tint)" },
  { key: "3_1_split", label: "3-1 split", tone: "var(--color-medium)", tint: "var(--color-medium-tint)" },
  { key: "2_2_split", label: "2-2 split", tone: "var(--color-high)", tint: "var(--color-high-tint)" },
];

export default function AgreementSummary({ summary, sampleSize }) {
  return (
    <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 14 }}>
      {ROWS.map(({ key, label, tone, tint }) => {
        const count = summary[key];
        const pct = ((count / sampleSize) * 100).toFixed(1);
        return (
          <div
            key={key}
            style={{
              background: tint,
              border: `1px solid ${tone}`,
              borderRadius: "var(--radius-lg)",
              padding: "18px 20px",
            }}
          >
            <div className="mono" style={{ fontSize: 28, fontWeight: 700, color: tone }}>{count}</div>
            <div style={{ fontSize: 13, color: "var(--color-text)", marginTop: 4 }}>{label}</div>
            <div style={{ fontSize: 11.5, color: "var(--color-text-muted)", marginTop: 2 }}>{pct}% of sample</div>
          </div>
        );
      })}
    </div>
  );
}
