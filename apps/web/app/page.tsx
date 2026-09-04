import Link from "next/link";

export default function HomePage() {
  return (
    <div className="page">
      <div className="page-kicker">ADFM Internal Platform</div>
      <h1 className="page-title">Analytics Command Center</h1>
      <p className="page-subtitle">
        The new in-house application shell. Streamlit remains available only as a legacy comparison layer while tools are migrated one by one into the web platform.
      </p>
      <div className="page-rule" />

      <div className="home-grid">
        <Link className="home-card live" href="/tools/rate-of-change">
          <div className="home-card-label">Live · Migrated</div>
          <div className="home-card-title">Rate of Change Regime Explorer</div>
          <div className="home-card-copy">
            Same Python calculation engine, served through FastAPI and rendered natively in the web application.
          </div>
        </Link>
        <div className="home-card">
          <div className="home-card-label">Next</div>
          <div className="home-card-title">Market Stress Composite</div>
          <div className="home-card-copy">Queued for the next page-by-page migration.</div>
        </div>
        <div className="home-card">
          <div className="home-card-label">Then</div>
          <div className="home-card-title">Equity Leadership & Rotation</div>
          <div className="home-card-copy">Will move after the ROC implementation is verified against the current app.</div>
        </div>
      </div>

      <div className="migration-note">
        Migration rule: a tool is only marked live here once its data path, controls, calculations and chart behavior work without opening or embedding Streamlit.
      </div>
    </div>
  );
}
