import { ArrowDownRight, ArrowUpRight, Sparkles } from "lucide-react";
import SignalBars from "./SignalBars";
import { formatPercent, isFlagged } from "../utils/risk";

function DetailRow({ label, value }) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between", padding: "9px 0", fontSize: 13.5 }}>
      <span style={{ color: "var(--color-text-muted)" }}>{label}</span>
      <span className="mono" style={{ fontWeight: 500 }}>{value}</span>
    </div>
  );
}

export default function CustomerDetail({ customer, meta }) {
  if (!customer) {
    return (
      <div style={{ padding: 32, color: "var(--color-text-muted)", fontSize: 14 }}>
        Select a customer from the list to view their risk profile.
      </div>
    );
  }

  const flagged = isFlagged(customer.predicted_proba, meta.decision_threshold);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      <div
        style={{
          background: "var(--color-surface)",
          border: "1px solid var(--color-border)",
          borderRadius: "var(--radius-lg)",
          boxShadow: "var(--shadow-card)",
          padding: 22,
        }}
      >
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: 12 }}>
          <div>
            <div style={{ fontSize: 12.5, color: "var(--color-text-muted)", fontWeight: 500 }}>Subscriber ID</div>
            <div className="mono" style={{ fontSize: 22, fontWeight: 600 }}>{customer.subscriber_id}</div>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
            <div style={{ textAlign: "right" }}>
              <div style={{ fontSize: 12.5, color: "var(--color-text-muted)", fontWeight: 500 }}>
                Predicted probability
              </div>
              <div className="mono" style={{ fontSize: 22, fontWeight: 600 }}>
                {formatPercent(customer.predicted_proba)}
              </div>
            </div>
            <SignalBars tier={customer.risk_tier} size="lg" showLabel />
          </div>
        </div>
      </div>

      <div
        style={{
          background: "var(--color-surface)",
          border: "1px solid var(--color-border)",
          borderRadius: "var(--radius-lg)",
          boxShadow: "var(--shadow-card)",
          padding: 22,
        }}
      >
        <h3 style={{ fontSize: 14.5, marginBottom: 4 }}>Prediction details</h3>
        <p style={{ fontSize: 12.5, color: "var(--color-text-muted)", margin: "0 0 6px" }}>
          Governance record for this prediction.
        </p>
        <div style={{ borderTop: "1px solid var(--color-border)" }}>
          <DetailRow label="Model" value={meta.model_used} />
          <DetailRow label="Model version" value={meta.model_version} />
          <DetailRow label="Snapshot / feature date" value={meta.snapshot_date} />
          <DetailRow label="Decision threshold" value={formatPercent(meta.decision_threshold)} />
          <DetailRow label="Predicted probability" value={formatPercent(customer.predicted_proba)} />
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "9px 0" }}>
            <span style={{ color: "var(--color-text-muted)", fontSize: 13.5 }}>Decision</span>
            <span
              style={{
                fontSize: 12.5,
                fontWeight: 600,
                padding: "4px 10px",
                borderRadius: 999,
                color: flagged ? "var(--color-brand)" : "var(--color-low)",
                background: flagged ? "var(--color-brand-tint)" : "var(--color-low-tint)",
              }}
            >
              {flagged ? "Flagged for retention" : "Not flagged"}
            </span>
          </div>
        </div>
      </div>

      <div
        style={{
          background: "var(--color-surface)",
          border: "1px solid var(--color-border)",
          borderRadius: "var(--radius-lg)",
          boxShadow: "var(--shadow-card)",
          padding: 22,
        }}
      >
        <h3 style={{ fontSize: 14.5, marginBottom: 12 }}>Why this customer is at risk</h3>
        <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
          {customer.top_factors.map((f, i) => {
            const up = f.direction === "increases_risk";
            const Icon = up ? ArrowUpRight : ArrowDownRight;
            return (
              <div key={i} style={{ display: "flex", alignItems: "flex-start", gap: 10 }}>
                <span
                  style={{
                    display: "inline-flex",
                    alignItems: "center",
                    justifyContent: "center",
                    width: 22,
                    height: 22,
                    borderRadius: "50%",
                    flexShrink: 0,
                    background: up ? "var(--color-high-tint)" : "var(--color-low-tint)",
                    color: up ? "var(--color-high)" : "var(--color-low)",
                  }}
                >
                  <Icon size={13} />
                </span>
                <span style={{ fontSize: 13.5, lineHeight: 1.5 }}>
                  {f.clause.charAt(0).toUpperCase() + f.clause.slice(1)}
                  <span style={{ color: "var(--color-text-muted)" }}>
                    {" "}
                    — {up ? "raises" : "lowers"} their estimated risk.
                  </span>
                </span>
              </div>
            );
          })}
        </div>
      </div>

      <div
        style={{
          background: "var(--color-surface)",
          border: "1px solid var(--color-border)",
          borderRadius: "var(--radius-lg)",
          boxShadow: "var(--shadow-card)",
          padding: 22,
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 12 }}>
          <h3 style={{ fontSize: 14.5 }}>Summary</h3>
          <span
            style={{
              display: "inline-flex",
              alignItems: "center",
              gap: 4,
              fontSize: 11,
              fontWeight: 500,
              color: "var(--color-text-muted)",
              border: "1px solid var(--color-border)",
              borderRadius: 999,
              padding: "2px 8px",
            }}
          >
            <Sparkles size={11} /> AI-generated summary
          </span>
        </div>
        <p style={{ fontSize: 14, lineHeight: 1.6, margin: 0 }}>{customer.explanation}</p>
      </div>
    </div>
  );
}
