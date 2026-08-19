import { useState } from "react";
import ThresholdSlider from "../components/ThresholdSlider";
import FlaggedCustomerList from "../components/FlaggedCustomerList";

export default function RetentionInsights({ data }) {
  const [threshold, setThreshold] = useState(data.meta.decision_threshold);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      <div>
        <h2 style={{ fontSize: 20 }}>Retention insights</h2>
        <p style={{ fontSize: 13.5, color: "var(--color-text-muted)", margin: "4px 0 0" }}>
          Adjust how aggressively to reach out, and see who that would include.
        </p>
      </div>

      <ThresholdSlider value={threshold} onChange={setThreshold} decisionThreshold={data.meta.decision_threshold} />
      <FlaggedCustomerList customers={data.customers} threshold={threshold} />
    </div>
  );
}
