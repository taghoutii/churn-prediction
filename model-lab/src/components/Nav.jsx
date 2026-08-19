import { GitCompare, ScatterChart, Users } from "lucide-react";

const PAGES = [
  { id: "comparison", label: "Full Model Comparison", icon: GitCompare },
  { id: "subscriber", label: "Subscriber Prediction", icon: Users },
  { id: "agreement", label: "Agreement Explorer", icon: ScatterChart },
];

export default function Nav({ page, onNavigate }) {
  return (
    <header
      style={{
        borderBottom: "1px solid var(--color-border)",
        background: "var(--color-surface)",
        position: "sticky",
        top: 0,
        zIndex: 10,
      }}
    >
      <div
        style={{
          maxWidth: 1280,
          margin: "0 auto",
          padding: "0 24px",
          display: "flex",
          alignItems: "center",
          gap: 32,
          height: 64,
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <span
            style={{
              width: 30,
              height: 30,
              borderRadius: "var(--radius-sm)",
              background: "var(--color-accent)",
              display: "inline-block",
            }}
            aria-hidden="true"
          />
          <div>
            <div style={{ fontWeight: 700, fontSize: 15, lineHeight: 1.2 }}>Model Lab</div>
            <div style={{ fontSize: 11, color: "var(--color-text-muted)", lineHeight: 1.2 }}>
              Churn model comparison
            </div>
          </div>
        </div>

        <nav style={{ display: "flex", gap: 4 }}>
          {PAGES.map(({ id, label, icon: Icon }) => {
            const active = page === id;
            return (
              <button
                key={id}
                onClick={() => onNavigate(id)}
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 7,
                  border: "none",
                  background: active ? "var(--color-accent-tint)" : "transparent",
                  color: active ? "var(--color-accent-dark)" : "var(--color-text-muted)",
                  fontWeight: active ? 600 : 500,
                  fontSize: 13.5,
                  padding: "8px 14px",
                  borderRadius: "var(--radius-md)",
                  cursor: "pointer",
                }}
              >
                <Icon size={16} />
                {label}
              </button>
            );
          })}
        </nav>
      </div>
    </header>
  );
}
