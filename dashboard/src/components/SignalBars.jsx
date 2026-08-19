import { RISK_COLORS } from "../utils/risk";

const SIZES = {
  sm: { width: 4, gap: 2, heights: [6, 9, 12] },
  md: { width: 5, gap: 3, heights: [8, 13, 18] },
  lg: { width: 6, gap: 3, heights: [10, 17, 24] },
};

/**
 * Phone-signal-strength style risk indicator: 1-3 filled bars, colored by
 * tier. Used everywhere a risk level is shown, instead of a plain dot.
 */
export default function SignalBars({ tier, size = "md", showLabel = false }) {
  const bars = tier === "high" ? 3 : tier === "medium" ? 2 : 1;
  const color = RISK_COLORS[tier] ?? RISK_COLORS.low;
  const { width, gap, heights } = SIZES[size] ?? SIZES.md;

  return (
    <span
      style={{ display: "inline-flex", alignItems: "center", gap: showLabel ? 8 : 0 }}
      role="img"
      aria-label={`${tier} risk`}
    >
      <span style={{ display: "inline-flex", alignItems: "flex-end", gap }}>
        {heights.map((h, i) => (
          <span
            key={i}
            style={{
              display: "inline-block",
              width,
              height: h,
              borderRadius: 1,
              background: i < bars ? color : "var(--color-border)",
            }}
          />
        ))}
      </span>
      {showLabel && (
        <span style={{ fontSize: 13, fontWeight: 500, color, textTransform: "capitalize" }}>
          {tier}
        </span>
      )}
    </span>
  );
}
