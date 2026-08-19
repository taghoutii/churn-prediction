import KpiCard from "../components/KpiCard";
import RiskDistributionChart from "../components/RiskDistributionChart";
import GlobalDriversChart from "../components/GlobalDriversChart";
import { formatPercent } from "../utils/risk";

export default function Overview({ data }) {
  const { overview, global_drivers, meta } = data;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>
      <div>
        <h2 style={{ fontSize: 20 }}>Portfolio overview</h2>
        <p style={{ fontSize: 13.5, color: "var(--color-text-muted)", margin: "4px 0 0" }}>
          Churn risk across the full evaluated customer base.
        </p>
      </div>

      <div style={{ display: "flex", gap: 16, flexWrap: "wrap" }}>
        <KpiCard label="Total customers evaluated" value={overview.total_customers.toLocaleString()} />
        <KpiCard
          label="Average churn probability"
          value={formatPercent(overview.avg_probability)}
          accent
        />
        <KpiCard label="Model in use" value={meta.model_used} sublabel={`version ${meta.model_version}`} />
        <KpiCard
          label="High-risk customers"
          value={overview.risk_tiers.high.toLocaleString()}
          sublabel={`${formatPercent(meta.risk_tier_bounds.medium_max, 0)}+ estimated risk`}
        />
      </div>

      <div style={{ display: "flex", gap: 20, flexWrap: "wrap" }}>
        <div
          style={{
            flex: "1 1 420px",
            background: "var(--color-surface)",
            border: "1px solid var(--color-border)",
            borderRadius: "var(--radius-lg)",
            boxShadow: "var(--shadow-card)",
            padding: 22,
          }}
        >
          <h3 style={{ fontSize: 14.5, marginBottom: 4 }}>Risk distribution</h3>
          <p style={{ fontSize: 12.5, color: "var(--color-text-muted)", margin: "0 0 12px" }}>
            Across all {overview.total_customers.toLocaleString()} evaluated customers.
          </p>
          <RiskDistributionChart riskTiers={overview.risk_tiers} totalCustomers={overview.total_customers} />
        </div>

        <div
          style={{
            flex: "1 1 420px",
            background: "var(--color-surface)",
            border: "1px solid var(--color-border)",
            borderRadius: "var(--radius-lg)",
            boxShadow: "var(--shadow-card)",
            padding: 22,
          }}
        >
          <h3 style={{ fontSize: 14.5, marginBottom: 4 }}>Most common risk drivers</h3>
          <p style={{ fontSize: 12.5, color: "var(--color-text-muted)", margin: "0 0 12px" }}>
            Based on the deployed model's own attributions for each customer, which can rank
            things differently than the analysis used earlier when deciding which factors to keep.
          </p>
          <GlobalDriversChart drivers={global_drivers} />
        </div>
      </div>
    </div>
  );
}
