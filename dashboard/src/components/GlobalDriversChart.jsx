import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

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
      <div style={{ fontWeight: 600 }}>{p.short_label}</div>
      <div className="mono">avg. impact {p.mean_abs_shap.toFixed(3)}</div>
      {p.note && (
        <div style={{ marginTop: 6, maxWidth: 220, color: "var(--color-text-muted)", fontWeight: 400 }}>
          {p.note}
        </div>
      )}
    </div>
  );
}

export default function GlobalDriversChart({ drivers }) {
  // `drivers` is already sorted descending (largest first) by the export
  // script. For a vertical-layout (horizontal bars) chart, recharts places
  // data[0] at the TOP of the category axis by default -- no reversal
  // needed. (Confirmed against recharts' own selectYAxisRange: for
  // layout="vertical" the Y range is [top, bottom], non-reversed.)
  const data = drivers;

  return (
    <ResponsiveContainer width="100%" height={Math.max(280, data.length * 34)}>
      <BarChart data={data} layout="vertical" margin={{ top: 8, right: 24, bottom: 8, left: 8 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" horizontal={false} />
        <XAxis type="number" tick={{ fontSize: 12, fill: "var(--color-text-muted)" }} axisLine={false} tickLine={false} />
        <YAxis
          type="category"
          dataKey="short_label"
          width={170}
          tick={{ fontSize: 12.5, fill: "var(--color-text)" }}
          axisLine={false}
          tickLine={false}
        />
        <Tooltip content={<CustomTooltip />} cursor={{ fill: "var(--color-surface-muted)" }} />
        <Bar dataKey="mean_abs_shap" radius={[0, 6, 6, 0]} fill="#E30613" maxBarSize={18} />
      </BarChart>
    </ResponsiveContainer>
  );
}
