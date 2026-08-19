import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";

// jsdom has no real text/layout measurement, so recharts silently skips
// rendering its SVG contents in tests (confirmed while investigating the
// reversed-order bug). Mocking recharts lets us assert the actual DATA
// ORDER handed to the chart -- which is what the bug was in -- without
// depending on pixel-level SVG rendering.
vi.mock("recharts", () => ({
  ResponsiveContainer: ({ children }) => <div>{children}</div>,
  BarChart: ({ data, children }) => (
    <div>
      <ul>
        {data.map((d) => (
          <li key={d.feature}>{d.short_label}</li>
        ))}
      </ul>
      {children}
    </div>
  ),
  Bar: () => null,
  XAxis: () => null,
  YAxis: () => null,
  CartesianGrid: () => null,
  Tooltip: () => null,
}));

import GlobalDriversChart from "../components/GlobalDriversChart";
import dashboardData from "../data/dashboardData.json";

describe("GlobalDriversChart ordering", () => {
  it("passes drivers through in the same (already-descending) order as the export, unreversed", () => {
    const drivers = dashboardData.global_drivers;
    render(<GlobalDriversChart drivers={drivers} />);

    const rendered = screen.getAllByRole("listitem").map((li) => li.textContent);
    const expected = drivers.map((d) => d.short_label);

    expect(rendered).toEqual(expected);
    // sanity: the exported data really is sorted descending
    const scores = drivers.map((d) => d.mean_abs_shap);
    expect(scores).toEqual([...scores].sort((a, b) => b - a));
  });

  it("puts the single largest driver first -- which recharts renders at the top for layout=\"vertical\"", () => {
    const drivers = dashboardData.global_drivers;
    render(<GlobalDriversChart drivers={drivers} />);
    const first = screen.getAllByRole("listitem")[0];
    const biggest = [...drivers].sort((a, b) => b.mean_abs_shap - a.mean_abs_shap)[0];
    expect(first.textContent).toBe(biggest.short_label);
  });
});
