import { LayoutDashboard, Users, Sliders } from "lucide-react";

const PAGES = [
  { id: "overview", label: "Overview", icon: LayoutDashboard },
  { id: "customers", label: "Customer Risk", icon: Users },
  { id: "insights", label: "Retention Insights", icon: Sliders },
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
          maxWidth: 1180,
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
              borderRadius: "50%",
              background: "var(--color-brand)",
              display: "inline-block",
            }}
            aria-hidden="true"
          />
          <span style={{ fontWeight: 700, fontSize: 16 }}>Ooredoo Churn Risk</span>
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
                  background: active ? "var(--color-brand-tint)" : "transparent",
                  color: active ? "var(--color-brand)" : "var(--color-text-muted)",
                  fontWeight: active ? 600 : 500,
                  fontSize: 14,
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
