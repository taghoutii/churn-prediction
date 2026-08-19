import { useState } from "react";
import SubscriberList from "../components/SubscriberList";
import ModelPredictionCard from "../components/ModelPredictionCard";
import AgreementIndicator from "../components/AgreementIndicator";

export default function SubscriberPrediction({ data }) {
  const { meta, customers } = data;
  const [selectedId, setSelectedId] = useState(customers[0]?.subscriber_id ?? null);
  const selected = customers.find((c) => c.subscriber_id === selectedId) ?? null;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      <div>
        <h2 style={{ fontSize: 20 }}>Subscriber multi-model prediction</h2>
        <p style={{ fontSize: 13.5, color: "var(--color-text-muted)", margin: "4px 0 0" }}>
          Same {customers.length} sampled subscribers as dashboard/, scored by all four models.
        </p>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "280px 1fr", gap: 20, alignItems: "start" }}>
        <div style={{ height: 640 }}>
          <SubscriberList customers={customers} selectedId={selectedId} onSelect={setSelectedId} />
        </div>

        {selected ? (
          <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
              <div>
                <div style={{ fontSize: 12, color: "var(--color-text-muted)" }}>Subscriber ID</div>
                <div className="mono" style={{ fontSize: 20, fontWeight: 700 }}>{selected.subscriber_id}</div>
              </div>
              <AgreementIndicator votesChurn={selected.votes_churn} allAgree={selected.all_agree} />
            </div>

            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(auto-fit, minmax(230px, 1fr))",
                gap: 14,
              }}
            >
              {meta.model_order.map((m) => (
                <ModelPredictionCard
                  key={m}
                  modelKey={m}
                  modelLabel={meta.model_labels[m]}
                  prediction={selected.predictions[m]}
                />
              ))}
            </div>
          </div>
        ) : (
          <div style={{ padding: 32, color: "var(--color-text-muted)", fontSize: 14 }}>
            Select a subscriber from the list.
          </div>
        )}
      </div>
    </div>
  );
}
