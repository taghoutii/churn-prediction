import { useState } from "react";
import Nav from "./components/Nav";
import FullComparison from "./pages/FullComparison";
import SubscriberPrediction from "./pages/SubscriberPrediction";
import AgreementExplorer from "./pages/AgreementExplorer";
import data from "./data/modelComparisonData.json";

const PAGE_COMPONENTS = {
  comparison: FullComparison,
  subscriber: SubscriberPrediction,
  agreement: AgreementExplorer,
};

export default function App() {
  const [page, setPage] = useState("comparison");
  const Page = PAGE_COMPONENTS[page];

  return (
    <div style={{ minHeight: "100vh", background: "var(--color-bg)" }}>
      <Nav page={page} onNavigate={setPage} />
      <main style={{ maxWidth: 1280, margin: "0 auto", padding: "28px 24px 64px" }}>
        <Page data={data} />
      </main>
    </div>
  );
}
