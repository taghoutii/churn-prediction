import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { RISK_COLORS, RISK_LABELS } from "../utils/risk";

function CustomTooltip({ active, payload }) {
  if (!active || !payload?.length) return null;
  const p = payload[0].payload;
  return (
    <div
      style={{
        background: "var(--color-surface)",
        border: "1px solid var(--color-border)",
        borderRadius: "var(--radius-sm)",
        padding: "8px 12px",
        fontSize: 13,
        boxShadow: "var(--shadow-card)",
      }}
    >
      <div style={{ fontWeight: 600 }}>{p.label}</div>
      <div className="mono">{p.count.toLocaleString()} customers ({p.pct.toFixed(1)}%)</div>
    </div>
  );
}

export default function RiskDistributionChart({ riskTiers, totalCustomers }) {
  const data = ["low", "medium", "high"].map((tier) => ({
    tier,
    label: RISK_LABELS[tier],
    count: riskTiers[tier] ?? 0,
    pct: totalCustomers ? ((riskTiers[tier] ?? 0) / totalCustomers) * 100 : 0,
  }));

  return (
    <ResponsiveContainer width="100%" height={260}>
      <BarChart data={data} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" vertical={false} />
        <XAxis dataKey="label" tick={{ fontSize: 13, fill: "var(--color-text-muted)" }} axisLine={{ stroke: "var(--color-border)" }} tickLine={false} />
        <YAxis tick={{ fontSize: 12, fill: "var(--color-text-muted)" }} axisLine={false} tickLine={false} />
        <Tooltip content={<CustomTooltip />} cursor={{ fill: "var(--color-surface-muted)" }} />
        <Bar dataKey="count" radius={[6, 6, 0, 0]} maxBarSize={80}>
          {data.map((d) => (
            <Cell key={d.tier} fill={RISK_COLORS[d.tier]} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
