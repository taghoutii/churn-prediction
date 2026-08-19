import Card from "../components/Card";
import AgreementSummary from "../components/AgreementSummary";
import DisagreementTable from "../components/DisagreementTable";

export default function AgreementExplorer({ data }) {
  const { meta, agreement_summary, xgb_lgbm_disagreements } = data;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      <div>
        <h2 style={{ fontSize: 20 }}>Model agreement explorer</h2>
        <p style={{ fontSize: 13.5, color: "var(--color-text-muted)", margin: "4px 0 0" }}>
          How often the four models' own-threshold predicted classes agree across the {meta.sample_size} sampled
          subscribers. This is exploratory, not a case for any one model.
        </p>
      </div>

      <Card title="Agreement across all 4 models">
        <AgreementSummary summary={agreement_summary} sampleSize={meta.sample_size} />
      </Card>

      <Card
        title="XGBoost vs. LightGBM disagreements"
        subtitle={`${xgb_lgbm_disagreements.length} of ${meta.sample_size} sampled subscribers -- these are the two leading tree-based candidates, so their disagreements get their own view. Sorted by probability gap, largest first.`}
      >
        <DisagreementTable disagreements={xgb_lgbm_disagreements} />
      </Card>
    </div>
  );
}
