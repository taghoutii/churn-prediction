import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import App from "../App";

// jsdom doesn't do layout, so recharts' ResponsiveContainer measures 0x0 by
// default. Give it a real size so charts actually mount their children.
beforeEach(() => {
  Object.defineProperty(HTMLElement.prototype, "offsetWidth", { configurable: true, value: 800 });
  Object.defineProperty(HTMLElement.prototype, "offsetHeight", { configurable: true, value: 400 });
  global.ResizeObserver = class {
    observe() {}
    unobserve() {}
    disconnect() {}
  };
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

function expectNoConsoleErrors(spy) {
  expect(spy).not.toHaveBeenCalled();
}

describe("model-lab pages render without console errors", () => {
  it("renders the Full Model Comparison page by default, with all four models in the metrics table", () => {
    const errorSpy = vi.spyOn(console, "error");
    render(<App />);

    expect(screen.getByText("Full model comparison")).toBeInTheDocument();
    expect(screen.getAllByText("Logistic Regression").length).toBeGreaterThan(0);
    expect(screen.getAllByText("XGBoost").length).toBeGreaterThan(0);
    expect(screen.getAllByText("LightGBM").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Voting Ensemble").length).toBeGreaterThan(0);
    expectNoConsoleErrors(errorSpy);
  });

  it("renders the Subscriber Prediction page with four side-by-side model cards and no network calls", () => {
    const errorSpy = vi.spyOn(console, "error");
    const fetchSpy = vi.spyOn(global, "fetch");
    render(<App />);

    fireEvent.click(screen.getByRole("button", { name: /subscriber prediction/i }));

    expect(screen.getByText("Subscriber multi-model prediction")).toBeInTheDocument();
    expect(screen.getByText(/of 4 models agree|All 4 models agree/)).toBeInTheDocument();
    expect(fetchSpy).not.toHaveBeenCalled();
    expectNoConsoleErrors(errorSpy);
  });

  it("renders the Model Agreement Explorer page", () => {
    const errorSpy = vi.spyOn(console, "error");
    render(<App />);

    fireEvent.click(screen.getByRole("button", { name: /agreement explorer/i }));

    expect(screen.getByText("Model agreement explorer")).toBeInTheDocument();
    expect(screen.getByText("XGBoost vs. LightGBM disagreements")).toBeInTheDocument();
    expectNoConsoleErrors(errorSpy);
  });
});
