"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const groups = [
  { name: "Equity Discovery", tools: ["ADFM Public Equities Baskets"] },
  {
    name: "Macro Regime",
    tools: [
      "Global Macro Regime",
      "Liquidity Conditions Monitor",
      "Yield Curve Rates Regime Monitor",
      "Credit Conditions Monitor",
      "Currency Tension Engine",
    ],
  },
  { name: "Equity Leadership", tools: ["Sector Breadth and Rotation", "Equity Leadership & Rotation"] },
  { name: "Fundamental Research", tools: ["ADFM Underwriter"] },
  {
    name: "Technical Confirmation",
    tools: [
      "ADFM Chart Terminal",
      "Cross-Asset Ratio Chartbook",
      "Rate of Change Regime Explorer",
      "Relative Volatility Lab",
    ],
  },
  {
    name: "Positioning + Flows",
    tools: [
      "ETF Flow Pressure Proxy",
      "Volume Based Sentiment Indicator",
      "Options Positioning Compass",
      "SEC 13F Exposure Browser",
      "CFTC Positioning Monitor",
    ],
  },
  {
    name: "Risk + Execution",
    tools: ["Market Stress Composite", "Catalyst Calendar", "Hedge Timer", "Position Sizing Lab"],
  },
  {
    name: "Historical Context",
    tools: ["Market Memory Explorer", "Monthly Seasonality Explorer", "Commodity Event Study"],
  },
];

const migratedRoutes: Record<string, string> = {
  "Rate of Change Regime Explorer": "/tools/rate-of-change",
};

export function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="sidebar">
      <div className="brand">
        <img
          className="brand-mark"
          src="https://raw.githubusercontent.com/adfundmgmt/ADFM-App-Folder/main/assets/ADFM_Logo_Naked.png"
          alt="ADFM"
        />
        <div>
          <div className="brand-name">AD Fund Management</div>
          <div className="brand-subtitle">Analytics</div>
        </div>
      </div>

      <nav className="sidebar-nav" aria-label="Analytics tools">
        <Link className={`nav-home ${pathname === "/" ? "active" : ""}`} href="/">
          Home
        </Link>

        {groups.map((group) => (
          <div className="nav-group" key={group.name}>
            <div className="nav-group-title">{group.name}</div>
            {group.tools.map((tool) => {
              const route = migratedRoutes[tool];
              if (!route) {
                return (
                  <div className="nav-tool disabled" key={tool} title="Pending migration">
                    <span>{tool}</span>
                    <span className="pending-dot" aria-hidden="true" />
                  </div>
                );
              }
              return (
                <Link
                  className={`nav-tool ${pathname === route ? "active" : ""}`}
                  href={route}
                  key={tool}
                >
                  <span>{tool}</span>
                  <span className="live-dot" aria-hidden="true" />
                </Link>
              );
            })}
          </div>
        ))}
      </nav>

      <div className="sidebar-footer">
        <span className="live-dot" />
        1 of 25 tools migrated
      </div>
    </aside>
  );
}
