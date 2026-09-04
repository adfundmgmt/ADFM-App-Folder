import { RocExplorer } from "../../../components/roc-explorer";

export default function RateOfChangePage() {
  return (
    <div className="page">
      <div className="page-kicker">ADFM Technical Confirmation</div>
      <h1 className="page-title">Rate of Change Regime Explorer</h1>
      <p className="page-subtitle">
        Track price trend, rate of change, acceleration and inflection points through the same Python calculation engine used by the legacy tool, now rendered directly in the ADFM web application.
      </p>
      <div className="page-rule" />
      <RocExplorer />
    </div>
  );
}
