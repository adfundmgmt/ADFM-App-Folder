"""ETF page rendering with real calculations and mocked external providers."""
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import streamlit as st
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def issuance_html():
    snapshot = json.loads((ROOT / 'data/ici/etf_net_issuance.json').read_text())
    labels = ['Domestic', 'World', 'Taxable', 'Municipal', 'Hybrid', 'Commodity', 'Total']
    html = '<p>Millions of dollars</p><table><tr><td></td>'
    html += ''.join(f'<td>{pd.Timestamp(week):%m/%d/%Y}</td>' for week in snapshot['weeks']) + '</tr>'
    for label, values in zip(labels, snapshot['values'], strict=True):
        html += f'<tr><td>{label}</td>' + ''.join(f'<td>{value * 1000:.0f}</td>' for value in values) + '</tr>'
    return html + '</table>'


def prices(tickers, **kwargs):
    del kwargs
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=800)
    columns = {}
    for i, ticker in enumerate(tickers):
        close = pd.Series(100 + np.arange(len(dates)) * .02, index=dates)
        for field, values in {'Open': close, 'High': close + 1, 'Low': close - 3,
                              'Close': close, 'Adj Close': close, 'Volume': (i + 1) * 100000.}.items():
            columns[field, ticker] = values
    frame = pd.DataFrame(columns, index=dates)
    frame.columns = pd.MultiIndex.from_tuples(frame.columns)
    return frame


class ETFPressurePageTests(unittest.TestCase):
    def setUp(self):
        st.cache_data.clear()

    def tearDown(self):
        st.cache_data.clear()

    def test_page_reports_industry_flows_and_bounded_individual_pressure(self):
        response = SimpleNamespace(text=issuance_html(), raise_for_status=lambda: None)
        with patch('requests.get', return_value=response), patch('adfm_core.market_data.download_market_data', side_effect=prices):
            app = AppTest.from_file(str(ROOT / 'pages/14_ETF_Flow_Pressure_Proxy.py')).run(timeout=30)
            self.assertEqual(list(app.exception), [])
            issuance = app.dataframe[0].value
            self.assertEqual(issuance.loc['Total'].iloc[0], 53.376)
            readings = next(d.value for d in app.dataframe if 'Pressure (%)' in d.value.columns)
            self.assertEqual(len(readings), 99)
            self.assertTrue(readings['Pressure (%)'].between(-100, 100).all())
            # Dollar volume changes with fund size; normalized pressure does not.
            self.assertAlmostEqual(readings['Pressure (%)'].min(), 50.)
            self.assertAlmostEqual(readings['Pressure (%)'].max(), 50.)
            self.assertEqual(len(app.metric), 0)
            self.assertEqual(len(app.warning), 0)
            app.selectbox[1].select('FX').run(timeout=30)
            self.assertEqual(list(app.exception), [])
            readings = next(d.value for d in app.dataframe if 'Pressure (%)' in d.value.columns)
            self.assertTrue(readings['Asset class'].eq('FX').all())

    def test_saved_issuance_keeps_its_original_week_when_provider_fails(self):
        with patch('requests.get', side_effect=TimeoutError), patch('adfm_core.market_data.download_market_data', return_value=pd.DataFrame()):
            app = AppTest.from_file(str(ROOT / 'pages/14_ETF_Flow_Pressure_Proxy.py')).run(timeout=30)
        self.assertEqual(list(app.exception), [])
        self.assertEqual(app.dataframe[0].value.columns[0], 'Sep 30, 2026')
        self.assertEqual(app.dataframe[0].value.loc['Total'].iloc[0], 53.376)
        self.assertEqual(len(app.get('plotly_chart')), 0)
        self.assertEqual(len(app.warning), 0)


if __name__ == '__main__':
    unittest.main()
