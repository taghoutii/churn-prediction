export default function Card({ title, subtitle, children, style }) {
  return (
    <div
      style={{
        background: "var(--color-surface)",
        border: "1px solid var(--color-border)",
        borderRadius: "var(--radius-lg)",
        boxShadow: "var(--shadow-card)",
        padding: 22,
        ...style,
      }}
    >
      {title && <h3 style={{ fontSize: 14.5, marginBottom: subtitle ? 4 : 12 }}>{title}</h3>}
      {subtitle && (
        <p style={{ fontSize: 12.5, color: "var(--color-text-muted)", margin: "0 0 12px" }}>{subtitle}</p>
      )}
      {children}
    </div>
  );
}
