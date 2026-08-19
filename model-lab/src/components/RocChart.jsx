import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { MODEL_COLORS } from "../utils/models";

const BASELINE = [{ fpr: 0, tpr: 0 }, { fpr: 1, tpr: 1 }];

export default function RocChart({ rocCurves, modelOrder, modelLabels, height = 320 }) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
        <XAxis
          dataKey="fpr"
          type="number"
          domain={[0, 1]}
          tick={{ fontSize: 12, fill: "var(--color-text-muted)" }}
          axisLine={{ stroke: "var(--color-border)" }}
          tickLine={false}
          label={{ value: "False positive rate", position: "insideBottom", offset: -4, fontSize: 12, fill: "var(--color-text-muted)" }}
        />
        <YAxis
          dataKey="tpr"
          type="number"
          domain={[0, 1]}
          tick={{ fontSize: 12, fill: "var(--color-text-muted)" }}
          axisLine={false}
          tickLine={false}
          label={{ value: "True positive rate", angle: -90, position: "insideLeft", fontSize: 12, fill: "var(--color-text-muted)" }}
        />
        <Tooltip
          formatter={(value) => value.toFixed(3)}
          contentStyle={{
            background: "var(--color-surface)",
            border: "1px solid var(--color-border)",
            borderRadius: "var(--radius-sm)",
            fontSize: 12.5,
          }}
        />
        <Legend wrapperStyle={{ fontSize: 12.5 }} formatter={(value) => modelLabels[value] ?? value} />
        <Line
          data={BASELINE}
          dataKey="tpr"
          name="baseline"
          stroke="var(--color-border)"
          strokeDasharray="4 4"
          dot={false}
          isAnimationActive={false}
          legendType="none"
        />
        {modelOrder.map((m) => (
          <Line
            key={m}
            data={rocCurves[m]}
            dataKey="tpr"
            name={m}
            stroke={MODEL_COLORS[m]}
            strokeWidth={2}
            dot={false}
            isAnimationActive={false}
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}
