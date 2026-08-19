import { formatPercent } from "../utils/models";

export default function DisagreementTable({ disagreements }) {
  if (disagreements.length === 0) {
    return (
      <div style={{ padding: 16, fontSize: 13.5, color: "var(--color-text-muted)" }}>
        XGBoost and LightGBM predicted the same class for every sampled subscriber.
      </div>
    );
  }

  return (
    <div style={{ overflowX: "auto" }}>
      <table style={{ fontSize: 13 }}>
        <thead>
          <tr style={{ borderBottom: "1px solid var(--color-border)" }}>
            {["Subscriber ID", "XGBoost proba", "XGBoost class", "LightGBM proba", "LightGBM class", "Gap"].map((h) => (
              <th key={h} style={{ textAlign: h === "Subscriber ID" ? "left" : "right", padding: "8px 12px", color: "var(--color-text-muted)", fontWeight: 500 }}>
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {disagreements.map((d) => (
            <tr key={d.subscriber_id} style={{ borderBottom: "1px solid var(--color-border)" }}>
              <td className="mono" style={{ padding: "9px 12px", fontWeight: 600 }}>{d.subscriber_id}</td>
              <td className="mono" style={{ padding: "9px 12px", textAlign: "right" }}>{formatPercent(d.xgboost_proba)}</td>
              <td style={{ padding: "9px 12px", textAlign: "right" }}>
                <span style={{ color: d.xgboost_class ? "var(--color-high)" : "var(--color-low)" }}>
                  {d.xgboost_class ? "Churn" : "No churn"}
                </span>
              </td>
              <td className="mono" style={{ padding: "9px 12px", textAlign: "right" }}>{formatPercent(d.lightgbm_proba)}</td>
              <td style={{ padding: "9px 12px", textAlign: "right" }}>
                <span style={{ color: d.lightgbm_class ? "var(--color-high)" : "var(--color-low)" }}>
                  {d.lightgbm_class ? "Churn" : "No churn"}
                </span>
              </td>
              <td className="mono" style={{ padding: "9px 12px", textAlign: "right", fontWeight: 600 }}>
                {formatPercent(d.gap)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
