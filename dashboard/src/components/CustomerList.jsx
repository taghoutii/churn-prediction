import { useState } from "react";
import { Search } from "lucide-react";
import SignalBars from "./SignalBars";
import { formatPercent } from "../utils/risk";

export default function CustomerList({ customers, selectedId, onSelect }) {
  const [query, setQuery] = useState("");

  const filtered = query.trim()
    ? customers.filter((c) => String(c.subscriber_id).includes(query.trim()))
    : customers;

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        border: "1px solid var(--color-border)",
        borderRadius: "var(--radius-lg)",
        background: "var(--color-surface)",
        boxShadow: "var(--shadow-card)",
        overflow: "hidden",
        height: "100%",
      }}
    >
      <div style={{ padding: 14, borderBottom: "1px solid var(--color-border)" }}>
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 8,
            border: "1px solid var(--color-border)",
            borderRadius: "var(--radius-sm)",
            padding: "7px 10px",
          }}
        >
          <Search size={15} color="var(--color-text-muted)" />
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search subscriber ID"
            aria-label="Search subscriber ID"
            style={{ border: "none", outline: "none", fontSize: 13.5, width: "100%" }}
          />
        </div>
      </div>

      <div style={{ overflowY: "auto", flex: 1 }}>
        {filtered.length === 0 && (
          <div style={{ padding: 16, fontSize: 13, color: "var(--color-text-muted)" }}>No matches.</div>
        )}
        {filtered.map((c) => {
          const active = c.subscriber_id === selectedId;
          return (
            <button
              key={c.subscriber_id}
              onClick={() => onSelect(c.subscriber_id)}
              style={{
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                width: "100%",
                textAlign: "left",
                padding: "11px 14px",
                border: "none",
                borderBottom: "1px solid var(--color-border)",
                background: active ? "var(--color-brand-tint)" : "transparent",
                cursor: "pointer",
              }}
            >
              <span className="mono" style={{ fontSize: 13, fontWeight: active ? 600 : 500 }}>
                {c.subscriber_id}
              </span>
              <span style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <span className="mono" style={{ fontSize: 12.5, color: "var(--color-text-muted)" }}>
                  {formatPercent(c.predicted_proba)}
                </span>
                <SignalBars tier={c.risk_tier} size="sm" />
              </span>
            </button>
          );
        })}
      </div>
    </div>
  );
}
