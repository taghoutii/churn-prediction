export default function KpiCard({ label, value, sublabel, accent = false }) {
  return (
    <div
      style={{
        background: "var(--color-surface)",
        border: "1px solid var(--color-border)",
        borderRadius: "var(--radius-lg)",
        boxShadow: "var(--shadow-card)",
        padding: "20px 22px",
        flex: "1 1 200px",
        minWidth: 200,
      }}
    >
      <div style={{ fontSize: 13, color: "var(--color-text-muted)", fontWeight: 500 }}>{label}</div>
      <div
        className="mono"
        style={{
          fontSize: 30,
          fontWeight: 600,
          marginTop: 8,
          color: accent ? "var(--color-brand)" : "var(--color-text)",
        }}
      >
        {value}
      </div>
      {sublabel && (
        <div style={{ fontSize: 12.5, color: "var(--color-text-muted)", marginTop: 6 }}>{sublabel}</div>
      )}
    </div>
  );
}
