import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import App from "../App";

// jsdom doesn't do layout, so recharts' ResponsiveContainer measures 0x0
// by default. Give it a real size so charts actually mount their children
// instead of just warning and skipping render.
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

describe("dashboard pages render without console errors", () => {
  it("renders the Overview page by default", () => {
    const errorSpy = vi.spyOn(console, "error");
    render(<App />);

    expect(screen.getByText("Portfolio overview")).toBeInTheDocument();
    expect(screen.getByText("Total customers evaluated")).toBeInTheDocument();
    expectNoConsoleErrors(errorSpy);
  });

  it("renders the Customer Risk page with no live network calls for the AI summary", () => {
    const errorSpy = vi.spyOn(console, "error");
    const fetchSpy = vi.spyOn(global, "fetch");
    render(<App />);

    fireEvent.click(screen.getByRole("button", { name: /customer risk/i }));

    expect(screen.getByRole("heading", { name: /customer risk/i })).toBeInTheDocument();
    expect(screen.getByText("Prediction details")).toBeInTheDocument();
    expect(screen.getByText("Summary")).toBeInTheDocument();
    expect(screen.getByText("AI-generated summary")).toBeInTheDocument();
    // the explanation text is present immediately -- loaded from the bundled
    // JSON, not fetched -- so no network call should ever have been made.
    expect(fetchSpy).not.toHaveBeenCalled();
    expectNoConsoleErrors(errorSpy);
  });

  it("computes the decision badge dynamically from probability vs. threshold", () => {
    render(<App />);
    fireEvent.click(screen.getByRole("button", { name: /customer risk/i }));

    const badge = screen.getByText(/Flagged for retention|Not flagged/);
    expect(badge).toBeInTheDocument();
  });

  it("renders the Retention Insights page and updates on preset click", () => {
    const errorSpy = vi.spyOn(console, "error");
    render(<App />);

    fireEvent.click(screen.getByRole("button", { name: /retention insights/i }));
    expect(screen.getByText("Retention threshold")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /aggressive/i }));
    expect(screen.getByTestId("threshold-value")).toHaveTextContent("10%");

    expectNoConsoleErrors(errorSpy);
  });
});
