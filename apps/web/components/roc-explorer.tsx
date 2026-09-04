"use client";

import dynamic from "next/dynamic";
import { FormEvent, useEffect, useMemo, useState } from "react";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

const API_BASE = process.env.NEXT_PUBLIC_ADFM_API_URL ?? "http://localhost:8000";

const timeframes = ["3M", "6M", "1Y", "3Y", "5Y", "10Y", "25Y", "Max"] as const;
const rocPeriods = ["10D", "20D", "63D", "126D", "252D"] as const;

type Timeframe = (typeof timeframes)[number];
type RocPeriod = (typeof rocPeriods)[number];
type ChartView = "Candlestick" | "Line";

type Point = {
  date: string;
  open: number | null;
  high: number | null;
  low: number | null;
  close: number | null;
  sma21: number | null;
  sma50: number | null;
  sma100: number | null;
  sma200: number | null;
  roc: number | null;
  rocSlope: number | null;
  acceleration: number | null;
  accelerationSlope: number | null;
  positiveInflection: boolean;
  negativeInflection: boolean;
};

type RocResponse = {
  tool: string;
  ticker: string;
  timeframe: Timeframe;
  rocPeriod: RocPeriod;
  source: string;
  dataThrough: string;
  dataQuality: string;
  observationCount: number;
  warning: string | null;
  latest: {
    close: number | null;
    roc: number | null;
    acceleration: number | null;
  };
  series: Point[];
};

function pct(value: number | null, digits = 2) {
  return value === null ? "—" : `${(value * 100).toFixed(digits)}%`;
}

function number(value: number | null) {
  return value === null ? "—" : value.toFixed(2);
}

export function RocExplorer() {
  const [draftTicker, setDraftTicker] = useState("^SPX");
  const [ticker, setTicker] = useState("^SPX");
  const [timeframe, setTimeframe] = useState<Timeframe>("3Y");
  const [rocPeriod, setRocPeriod] = useState<RocPeriod>("63D");
  const [chartView, setChartView] = useState<ChartView>("Candlestick");
  const [showInflections, setShowInflections] = useState(true);
  const [data, setData] = useState<RocResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    const params = new URLSearchParams({ ticker, timeframe, roc_period: rocPeriod });

    setLoading(true);
    setError(null);

    fetch(`${API_BASE}/api/tools/rate-of-change?${params.toString()}`, { signal: controller.signal })
      .then(async (response) => {
        if (!response.ok) {
          const payload = await response.json().catch(() => null);
          throw new Error(payload?.detail ?? `Request failed with ${response.status}`);
        }
        return response.json() as Promise<RocResponse>;
      })
      .then(setData)
      .catch((reason: unknown) => {
        if (reason instanceof DOMException && reason.name === "AbortError") return;
        setError(reason instanceof Error ? reason.message : "Unable to load ROC data.");
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });

    return () => controller.abort();
  }, [ticker, timeframe, rocPeriod]);

  function submitTicker(event: FormEvent) {
    event.preventDefault();
    const next = draftTicker.trim().toUpperCase();
    if (next) setTicker(next);
  }

  const figure = useMemo(() => {
    if (!data) return null;

    const x = data.series.map((point) => point.date);
    const traces: any[] = [];

    if (chartView === "Candlestick") {
      traces.push({
        type: "candlestick",
        x,
        open: data.series.map((point) => point.open),
        high: data.series.map((point) => point.high),
        low: data.series.map((point) => point.low),
        close: data.series.map((point) => point.close),
        name: "Price",
        xaxis: "x",
        yaxis: "y",
        increasing: { line: { color: "#58745e" }, fillcolor: "rgba(88,116,94,.48)" },
        decreasing: { line: { color: "#9a5e5e" }, fillcolor: "rgba(154,94,94,.48)" },
        hovertemplate:
          "%{x}<br>Open: %{open:.2f}<br>High: %{high:.2f}<br>Low: %{low:.2f}<br>Close: %{close:.2f}<extra></extra>",
      });
    } else {
      traces.push({
        type: "scatter",
        mode: "lines",
        x,
        y: data.series.map((point) => point.close),
        name: "Price",
        xaxis: "x",
        yaxis: "y",
        line: { color: "#171715", width: 2.1 },
        hovertemplate: "%{x}<br>Close: %{y:.2f}<extra></extra>",
      });
    }

    const averages = [
      ["SMA 21", "sma21", "#58769c"],
      ["SMA 50", "sma50", "#bd7966"],
      ["SMA 100", "sma100", "#8177a2"],
      ["SMA 200", "sma200", "#61758c"],
    ] as const;

    for (const [name, key, color] of averages) {
      traces.push({
        type: "scatter",
        mode: "lines",
        x,
        y: data.series.map((point) => point[key]),
        name,
        xaxis: "x",
        yaxis: "y",
        line: { color, width: 1.45 },
        hovertemplate: `%{x}<br>${name}: %{y:.2f}<extra></extra>`,
      });
    }

    traces.push({
      type: "scatter",
      mode: "lines",
      x,
      y: data.series.map((point) => point.roc),
      name: `ROC ${rocPeriod}`,
      xaxis: "x2",
      yaxis: "y2",
      line: { color: "#58769c", width: 2 },
      hovertemplate: "%{x}<br>ROC: %{y:.2%}<extra></extra>",
    });

    traces.push({
      type: "scatter",
      mode: "lines",
      x,
      y: data.series.map((point) => point.acceleration),
      name: "Acceleration",
      xaxis: "x3",
      yaxis: "y3",
      line: { color: "#8177a2", width: 1.9 },
      hovertemplate: "%{x}<br>Acceleration: %{y:.4%}<extra></extra>",
    });

    if (showInflections) {
      const positive = data.series.filter((point) => point.positiveInflection);
      const negative = data.series.filter((point) => point.negativeInflection);
      traces.push({
        type: "scatter",
        mode: "markers",
        x: positive.map((point) => point.date),
        y: positive.map((point) => point.acceleration),
        name: "Positive inflection",
        xaxis: "x3",
        yaxis: "y3",
        marker: { symbol: "triangle-up", size: 8, color: "#58745e" },
        hovertemplate: "%{x}<br>Positive inflection<extra></extra>",
      });
      traces.push({
        type: "scatter",
        mode: "markers",
        x: negative.map((point) => point.date),
        y: negative.map((point) => point.acceleration),
        name: "Negative inflection",
        xaxis: "x3",
        yaxis: "y3",
        marker: { symbol: "triangle-down", size: 8, color: "#9a5e5e" },
        hovertemplate: "%{x}<br>Negative inflection<extra></extra>",
      });
    }

    const axisCommon = {
      showgrid: true,
      gridcolor: "#ece8e1",
      zeroline: false,
      linecolor: "#bdb7ad",
      tickcolor: "#bdb7ad",
      tickfont: { color: "#6f6c65", size: 10 },
      fixedrange: false,
    };

    return {
      data: traces,
      layout: {
        autosize: true,
        paper_bgcolor: "#ffffff",
        plot_bgcolor: "#ffffff",
        margin: { l: 66, r: 28, t: 30, b: 50 },
        hovermode: "x unified",
        dragmode: "zoom",
        legend: {
          orientation: "h",
          x: 0,
          y: 1.03,
          font: { size: 10, color: "#55514a" },
          bgcolor: "rgba(255,255,255,0)",
        },
        xaxis: { ...axisCommon, anchor: "y", showticklabels: false, rangeslider: { visible: false } },
        xaxis2: { ...axisCommon, anchor: "y2", matches: "x", showticklabels: false },
        xaxis3: { ...axisCommon, anchor: "y3", matches: "x", showticklabels: true },
        yaxis: { ...axisCommon, domain: [0.47, 1], title: { text: "Price", font: { size: 10 } } },
        yaxis2: {
          ...axisCommon,
          domain: [0.235, 0.41],
          tickformat: ".1%",
          title: { text: `ROC ${rocPeriod}`, font: { size: 10 } },
        },
        yaxis3: {
          ...axisCommon,
          domain: [0, 0.175],
          tickformat: ".2%",
          title: { text: "Acceleration", font: { size: 10 } },
        },
        shapes: [
          { type: "line", xref: "paper", x0: 0, x1: 1, yref: "y2", y0: 0, y1: 0, line: { color: "#9c978e", width: 1 } },
          { type: "line", xref: "paper", x0: 0, x1: 1, yref: "y3", y0: 0, y1: 0, line: { color: "#9c978e", width: 1 } },
        ],
      },
    };
  }, [data, chartView, rocPeriod, showInflections]);

  return (
    <>
      <form className="control-bar" onSubmit={submitTicker}>
        <div className="field">
          <label htmlFor="ticker">Ticker symbol</label>
          <input
            id="ticker"
            value={draftTicker}
            onChange={(event) => setDraftTicker(event.target.value)}
            onBlur={() => {
              const next = draftTicker.trim().toUpperCase();
              if (next) setTicker(next);
            }}
            aria-label="Ticker symbol"
          />
        </div>
        <div className="field">
          <label htmlFor="timeframe">Analysis window</label>
          <select id="timeframe" value={timeframe} onChange={(event) => setTimeframe(event.target.value as Timeframe)}>
            {timeframes.map((item) => <option key={item}>{item}</option>)}
          </select>
        </div>
        <div className="field">
          <label htmlFor="roc-period">ROC period</label>
          <select id="roc-period" value={rocPeriod} onChange={(event) => setRocPeriod(event.target.value as RocPeriod)}>
            {rocPeriods.map((item) => <option key={item}>{item}</option>)}
          </select>
        </div>
        <div className="field">
          <label htmlFor="chart-view">Chart view</label>
          <select id="chart-view" value={chartView} onChange={(event) => setChartView(event.target.value as ChartView)}>
            <option>Candlestick</option>
            <option>Line</option>
          </select>
        </div>
        <label className="toggle-wrap">
          <input type="checkbox" checked={showInflections} onChange={(event) => setShowInflections(event.target.checked)} />
          Inflections
        </label>
      </form>

      {error && <div className="error">{error}</div>}
      {data && !error && (
        <>
          <div className="status-line">
            <span><strong>{data.ticker}</strong></span>
            <span>Data through <strong>{data.dataThrough}</strong></span>
            <span>Source <strong>{data.source}</strong></span>
            <span>Quality <strong>{data.dataQuality}</strong></span>
            <span><strong>{data.observationCount}</strong> calculated observations</span>
          </div>
          {data.warning && <div className="alert">{data.warning}</div>}
          <div className="metric-row">
            <div className="metric">
              <div className="metric-label">Latest Close</div>
              <div className="metric-value">{number(data.latest.close)}</div>
            </div>
            <div className="metric">
              <div className="metric-label">ROC {data.rocPeriod}</div>
              <div className="metric-value">{pct(data.latest.roc)}</div>
            </div>
            <div className="metric">
              <div className="metric-label">Acceleration</div>
              <div className="metric-value">{pct(data.latest.acceleration, 3)}</div>
            </div>
          </div>
        </>
      )}

      <div className="chart-panel">
        {loading || !figure ? (
          <div className="chart-loading">Loading completed-session market data…</div>
        ) : (
          <Plot
            data={figure.data}
            layout={figure.layout as any}
            config={{ responsive: true, displaylogo: false, scrollZoom: true }}
            style={{ width: "100%", height: "700px" }}
            useResizeHandler
          />
        )}
      </div>
    </>
  );
}
