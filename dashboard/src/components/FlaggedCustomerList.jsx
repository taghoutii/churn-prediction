import SignalBars from "./SignalBars";
import { formatPercent } from "../utils/risk";

export default function FlaggedCustomerList({ customers, threshold }) {
  const flagged = customers
    .filter((c) => c.predicted_proba >= threshold)
    .sort((a, b) => b.predicted_proba - a.predicted_proba);

  return (
    <div
      style={{
        background: "var(--color-surface)",
        border: "1px solid var(--color-border)",
        borderRadius: "var(--radius-lg)",
        boxShadow: "var(--shadow-card)",
        overflow: "hidden",
      }}
    >
      <div style={{ padding: "16px 22px", borderBottom: "1px solid var(--color-border)" }}>
        <h3 style={{ fontSize: 14.5 }}>
          Customers at or above threshold{" "}
          <span className="mono" style={{ color: "var(--color-text-muted)", fontWeight: 400, fontSize: 13 }}>
            ({flagged.length} of {customers.length} sampled)
          </span>
        </h3>
      </div>

      {flagged.length === 0 ? (
        <div style={{ padding: 22, fontSize: 13.5, color: "var(--color-text-muted)" }}>
          No sampled customers meet this threshold.
        </div>
      ) : (
        <div>
          {flagged.map((c) => {
            const topFactor = c.top_factors.find((f) => f.direction === "increases_risk") ?? c.top_factors[0];
            return (
              <div
                key={c.subscriber_id}
                style={{
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "space-between",
                  gap: 16,
                  padding: "13px 22px",
                  borderBottom: "1px solid var(--color-border)",
                }}
              >
                <div style={{ display: "flex", alignItems: "center", gap: 14, minWidth: 0 }}>
                  <SignalBars tier={c.risk_tier} size="sm" />
                  <span className="mono" style={{ fontSize: 13.5, fontWeight: 600 }}>{c.subscriber_id}</span>
                  <span
                    style={{
                      fontSize: 13,
                      color: "var(--color-text-muted)",
                      overflow: "hidden",
                      textOverflow: "ellipsis",
                      whiteSpace: "nowrap",
                    }}
                  >
                    {topFactor?.clause}
                  </span>
                </div>
                <span className="mono" style={{ fontSize: 13.5, fontWeight: 600, flexShrink: 0 }}>
                  {formatPercent(c.predicted_proba)}
                </span>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
