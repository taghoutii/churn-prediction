import { useState } from "react";
import { Search } from "lucide-react";

export default function SubscriberList({ customers, selectedId, onSelect }) {
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
                padding: "10px 14px",
                border: "none",
                borderBottom: "1px solid var(--color-border)",
                background: active ? "var(--color-accent-tint)" : "transparent",
                cursor: "pointer",
              }}
            >
              <span className="mono" style={{ fontSize: 13, fontWeight: active ? 600 : 500 }}>
                {c.subscriber_id}
              </span>
              <span
                className="mono"
                style={{
                  fontSize: 11,
                  fontWeight: 600,
                  padding: "2px 7px",
                  borderRadius: 999,
                  color: c.all_agree ? "var(--color-low)" : "var(--color-medium)",
                  background: c.all_agree ? "var(--color-low-tint)" : "var(--color-medium-tint)",
                }}
              >
                {c.votes_churn}/4
              </span>
            </button>
          );
        })}
      </div>
    </div>
  );
}
