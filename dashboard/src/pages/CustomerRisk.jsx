import { useMemo, useState } from "react";
import CustomerList from "../components/CustomerList";
import CustomerDetail from "../components/CustomerDetail";

export default function CustomerRisk({ data }) {
  const sorted = useMemo(
    () => [...data.customers].sort((a, b) => b.predicted_proba - a.predicted_proba),
    [data.customers]
  );
  const [selectedId, setSelectedId] = useState(sorted[0]?.subscriber_id ?? null);

  const selected = sorted.find((c) => c.subscriber_id === selectedId) ?? null;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      <div>
        <h2 style={{ fontSize: 20 }}>Customer risk</h2>
        <p style={{ fontSize: 13.5, color: "var(--color-text-muted)", margin: "4px 0 0" }}>
          Sampled customers ({sorted.length}), sorted by estimated churn risk.
        </p>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "300px 1fr", gap: 20, alignItems: "start" }}>
        <div style={{ height: 640 }}>
          <CustomerList customers={sorted} selectedId={selectedId} onSelect={setSelectedId} />
        </div>
        <CustomerDetail customer={selected} meta={data.meta} />
      </div>
    </div>
  );
}
