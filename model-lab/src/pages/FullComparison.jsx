import Card from "../components/Card";
import MetricsTable from "../components/MetricsTable";
import RocChart from "../components/RocChart";
import PrChart from "../components/PrChart";
import ConfusionMatrix from "../components/ConfusionMatrix";

export default function FullComparison({ data }) {
  const { meta, metrics, roc_curves, pr_curves, confusion_matrices } = data;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      <div>
        <h2 style={{ fontSize: 20 }}>Full model comparison</h2>
        <p style={{ fontSize: 13.5, color: "var(--color-text-muted)", margin: "4px 0 0" }}>
          Logistic Regression, XGBoost, LightGBM, and the Voting Ensemble, evaluated on the same{" "}
          {meta.test_set_size.toLocaleString()}-row held-out test set. This is a straight comparison, not a
          recommendation -- see the per-metric marks below.
        </p>
      </div>

      <Card title="Metrics (standard 0.5 decision cutoff)" subtitle="Best value in each row is marked. No model wins every metric.">
        <MetricsTable metrics={metrics} modelOrder={meta.model_order} modelLabels={meta.model_labels} />
      </Card>

      <Card
        title="Precision-recall curves"
        subtitle="Prioritized over ROC here: with true churn at ~8.9% of the test set, PR-AUC is the more informative summary of a large class imbalance than ROC-AUC, which can look optimistic even for a weak model on rare-positive data."
      >
        <PrChart prCurves={pr_curves} modelOrder={meta.model_order} modelLabels={meta.model_labels} height={420} />
      </Card>

      <Card title="ROC curves">
        <RocChart rocCurves={roc_curves} modelOrder={meta.model_order} modelLabels={meta.model_labels} height={320} />
      </Card>

      <Card
        title="Confusion matrices"
        subtitle={`Each model at its own 5:1 (missed-churner : false-alarm) cost-optimal threshold -- not a shared 0.5 cutoff.`}
      >
        <div style={{ display: "grid", gridTemplateColumns: "repeat(2, 1fr)", gap: 16 }}>
          {meta.model_order.map((m) => (
            <ConfusionMatrix key={m} modelKey={m} modelLabel={meta.model_labels[m]} matrix={confusion_matrices[m]} />
          ))}
        </div>
      </Card>
    </div>
  );
}
