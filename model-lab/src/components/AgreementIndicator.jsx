export default function AgreementIndicator({ votesChurn, allAgree }) {
  const label = allAgree
    ? `All 4 models agree (${votesChurn === 4 ? "churn" : "no churn"})`
    : `${Math.max(votesChurn, 4 - votesChurn)} of 4 models agree`;

  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: 10,
        padding: "10px 16px",
        borderRadius: "var(--radius-md)",
        background: allAgree ? "var(--color-low-tint)" : "var(--color-medium-tint)",
        border: `1px solid ${allAgree ? "var(--color-low)" : "var(--color-medium)"}`,
      }}
    >
      <span
        className="mono"
        style={{
          fontSize: 13,
          fontWeight: 700,
          color: allAgree ? "var(--color-low)" : "var(--color-medium)",
        }}
      >
        {votesChurn}/4
      </span>
      <span style={{ fontSize: 13, color: "var(--color-text)" }}>{label}</span>
    </div>
  );
}
