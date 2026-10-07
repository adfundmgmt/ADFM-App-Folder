"""Checks for normalized ETF pressure and published category net issuance."""
import unittest

import numpy as np
import pandas as pd

from adfm_core.etf_pressure import parse_ici_issuance, pressure_reading


def bars():
    dates = pd.bdate_range('2026-01-02', periods=90)
    frame = pd.DataFrame({'Open': 100., 'High': 110., 'Low': 90.,
                          'Close': 105., 'Adj Close': 105., 'Volume': 100.}, index=dates)
    frame.loc[dates[-5:], 'Volume'] = 200.
    return frame, dates


class ETFPressureTests(unittest.TestCase):
    def test_pressure_is_bounded_and_independent_of_fund_size(self):
        frame, dates = bars()
        value = pressure_reading(frame, dates, 5)
        self.assertAlmostEqual(value['Pressure (%)'], 50.)
        self.assertAlmostEqual(value['Activity (x)'], 2.)
        large = frame.copy()
        large['Volume'] *= 1000
        self.assertAlmostEqual(pressure_reading(large, dates, 5)['Pressure (%)'], 50.)

    def test_missing_volume_is_not_neutral_pressure(self):
        frame, dates = bars()
        frame.loc[dates[-2], 'Volume'] = np.nan
        result = pressure_reading(frame, dates, 5)
        self.assertTrue(np.isnan(result['Pressure (%)']))
        self.assertEqual(result['Status'], 'Incomplete window')

    def test_stale_endpoint_is_not_ranked_as_current(self):
        frame, dates = bars()
        result = pressure_reading(frame.iloc[:-1], dates, 5)
        self.assertTrue(np.isnan(result['Pressure (%)']))

    def test_adjusted_returns_prevent_dividends_from_becoming_selling_volume(self):
        frame, dates = bars()
        frame['Close'] = 100.
        frame['Adj Close'] = 100.
        frame.loc[dates[-5:], 'Close'] = 99.
        result = pressure_reading(frame, dates, 5)
        self.assertEqual(result['Return (%)'], 0.)
        self.assertEqual(result['Up/down volume (%)'], 0.)

    def test_window_return_includes_first_session_and_prior_pressure_is_separate(self):
        frame, dates = bars()
        frame.loc[dates[-5:], ['Close', 'Adj Close']] = 110.
        result = pressure_reading(frame, dates, 5)
        self.assertAlmostEqual(result['Return (%)'], (110 / 105 - 1) * 100)
        self.assertAlmostEqual(result['Pressure (%)'], 100.)
        self.assertAlmostEqual(result['Pressure change (pp)'], 50.)

    def test_invalid_range_is_not_silently_clipped_into_a_strong_signal(self):
        frame, dates = bars()
        frame.loc[dates[-1], 'Close'] = 200.
        self.assertTrue(np.isnan(pressure_reading(frame, dates, 5)['Pressure (%)']))

    def test_future_observations_do_not_change_a_historical_reading(self):
        frame, dates = bars()
        result = pressure_reading(frame, dates[:-5], 5)
        truncated = pressure_reading(frame.iloc[:-5], dates[:-5], 5)
        self.assertEqual(result, truncated)

    def test_issuance_keeps_categories_separate_from_subtotals_and_converts_to_billions(self):
        rows = {'Equity': [30, 20], 'Domestic': [20, 15], 'World': [10, 5],
                'Hybrid': [1, 1], 'Bond': [12, 10], 'Taxable': [10, 8],
                'Municipal': [2, 2], 'Commodity': [-3, -1], 'Total': [40, 30]}
        html = '<p>Millions of dollars</p><table><tr><td></td><td>2/4/2026</td><td>1/28/2026</td></tr>'
        for label, values in rows.items():
            html += '<tr><td>' + label + '</td>' + ''.join(f'<td>{v * 1000:,}</td>' for v in values) + '</tr>'
        parsed = parse_ici_issuance(html + '</table>')
        self.assertEqual(parsed.index.tolist(), ['US equity', 'International equity', 'Taxable bonds', 'Municipal bonds', 'Hybrid', 'Commodities', 'Total'])
        self.assertEqual(parsed.loc['Total'].iloc[0], 40.)
        self.assertEqual(parsed.loc['Commodities'].iloc[0], -3.)
        self.assertEqual(parsed.iloc[:-1, 0].sum(), parsed.iloc[-1, 0])

    def test_issuance_rejects_a_changed_or_inconsistent_table(self):
        with self.assertRaises(ValueError):
            parse_ici_issuance('<table><tr><td>Domestic</td><td>100</td></tr></table>')


if __name__ == '__main__':
    unittest.main()
