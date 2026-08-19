import { useState } from "react";
import Nav from "./components/Nav";
import Overview from "./pages/Overview";
import CustomerRisk from "./pages/CustomerRisk";
import RetentionInsights from "./pages/RetentionInsights";
import dashboardData from "./data/dashboardData.json";

const PAGE_COMPONENTS = {
  overview: Overview,
  customers: CustomerRisk,
  insights: RetentionInsights,
};

export default function App() {
  const [page, setPage] = useState("overview");
  const Page = PAGE_COMPONENTS[page];

  return (
    <div style={{ minHeight: "100vh", background: "var(--color-bg)" }}>
      <Nav page={page} onNavigate={setPage} />
      <main style={{ maxWidth: 1180, margin: "0 auto", padding: "28px 24px 64px" }}>
        <Page data={dashboardData} />
      </main>
    </div>
  );
}
