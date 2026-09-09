"""Offline regression tests. Run with unittest; never executes the Streamlit page."""
import ast
import unittest
import math
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from zoneinfo import ZoneInfo
from html import escape
import numpy as np
import pandas as pd
import exchange_calendars as xcals

SOURCE = Path(__file__).resolve().parents[1] / 'pages/1_ADFM_Public_Equities_Baskets.py'
if not SOURCE.exists():
    SOURCE = Path(__file__).with_name('integrity_fixed.py')
tree = ast.parse(SOURCE.read_text())
scope = dict(globals())
scope.update(MIN_DAILY_MEMBER_COVERAGE=.6, MIN_LIVE_MEMBER_COVERAGE=.5,
             MAX_FORWARD_FILL_SESSIONS=5, BASKET_KEY_SEPARATOR=' :: ',
             BENCH='SPY', NY_TZ=ZoneInfo('America/New_York'))
for node in tree.body:
    target = getattr(node, 'target', None)
    if isinstance(node, ast.Assign):
        target = node.targets[0]
    if getattr(target, 'id', '') in {'FUND_SYMBOLS','LOCAL_CALENDARS','FX_CONVERSIONS','FUND_QUOTE_TYPES','CATEGORIES'}:
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(SOURCE),'exec'),scope)
for node in tree.body:
    if isinstance(node, ast.FunctionDef):
        node.decorator_list = []
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(SOURCE),'exec'),scope)

class IntegrityTests(unittest.TestCase):
    def test_gap_not_compounded(self):
        i=pd.bdate_range('2026-01-05',periods=4)
        p=pd.DataFrame({'A':[100,110,121,121],'B':[100,np.nan,100,100]},index=i)
        r=scope['ew_rets_from_levels'](p,{'x':['A','B']})['x']
        self.assertTrue(pd.isna(r.iloc[1]) and pd.isna(r.iloc[2]))
        self.assertTrue(pd.isna(scope['pct_since'](100*(1+r).cumprod(),i[0])))
        self.assertTrue(scope['cumulative_return_since'](r,i[0]).empty)

    def test_partial_member_gap_keeps_basket_when_coverage_is_sufficient(self):
        i = pd.bdate_range('2026-01-05', periods=5)
        p = pd.DataFrame({
            'A': [100, 101, 102, 103, 104],
            'B': [100, 100, 101, 102, 103],
            'C': [100, np.nan, np.nan, 103, 104],
        }, index=i)
        r = scope['ew_rets_from_levels'](p, {'x': ['A', 'B', 'C']})['x']
        self.assertTrue(pd.notna(r.iloc[1]))
        self.assertTrue(pd.notna(r.iloc[2]))

    def test_indicator_settings_are_dynamic_and_bounded(self):
        as_of = pd.Timestamp('2026-09-09')
        macd_1m = scope['macd_settings']('1M', as_of)
        macd_5y = scope['macd_settings']('5Y', as_of)
        ema_1m = scope['ema_settings']('1M', as_of)
        ema_5y = scope['ema_settings']('5Y', as_of)
        self.assertNotEqual(macd_1m[:3], macd_5y[:3])
        self.assertNotEqual(ema_1m, ema_5y)
        self.assertLess(macd_5y[1], 150)
        self.assertLess(ema_5y[2], 100)

    def test_clean_reference(self):
        i=pd.bdate_range('2026-01-05',periods=3)
        p=pd.DataFrame({'A':[100,110,121],'B':[100,100,100]},index=i)
        r=scope['ew_rets_from_levels'](p,{'x':['A','B']})['x']
        self.assertAlmostEqual(scope['pct_since'](100*(1+r).cumprod(),i[0]),.1025)

    def test_common_endpoints(self):
        i=pd.bdate_range('2026-01-05',periods=4)
        for x in ([100,101,102,np.nan],[100,np.nan,102,103],[np.nan,101,102,103]):
            self.assertTrue(pd.isna(scope['pct_since'](pd.Series(x,index=i),i[0])))

    def test_indicators_do_not_compress_gaps(self):
        s=pd.Series([100.]*60+[np.nan]+[101.]*10)
        self.assertTrue(pd.isna(scope['basket_vs_dma_pct'](s,21)))
        self.assertEqual(scope['ema_regime'](s),'N/A')
        self.assertEqual(scope['horizon_macd_momentum'](s,(12,26,9,5,63)),'N/A')

    def test_equity_breadth(self):
        i=pd.to_datetime(['2026-01-02','2026-02-02'])
        p=pd.DataFrame({'A':[100,110],'B':[100,100],'SPY':[100,120]},index=i)
        r=scope['compute_basket_breadth'](p,{'proxy':['SPY'],'mixed':['SPY','A','B'],'low':['A','B','C','D']},i[0])
        self.assertTrue(pd.isna(r['proxy']) and pd.isna(r['low']))
        self.assertEqual(r['mixed'],50)

    def test_early_close(self):
        f=scope['completed_us_sessions']
        self.assertEqual(f('2026-11-23',pd.Timestamp('2026-11-27 13:16',tz='America/New_York'))[-1],pd.Timestamp('2026-11-27'))
        self.assertEqual(f('2026-11-23',pd.Timestamp('2026-11-27 13:10',tz='America/New_York'))[-1],pd.Timestamp('2026-11-25'))

    def test_holiday_carry_not_feed_gap(self):
        i=pd.to_datetime(['2026-04-30','2026-05-01','2026-05-04'])
        p=pd.DataFrame({'ABC.PA':[100,np.nan,102]},index=i)
        self.assertEqual(scope['carry_exchange_closures'](p).iloc[1,0],100)
        p.iloc[0,0]=np.nan
        self.assertTrue(pd.isna(scope['carry_exchange_closures'](p).iloc[1,0]))
        q=pd.DataFrame({'ABC.PA':[100,np.nan,102]},index=pd.bdate_range('2026-05-04',periods=3))
        self.assertTrue(pd.isna(scope['carry_exchange_closures'](q).iloc[1,0]))

    def test_sanitization(self):
        x=scope['_to_float_frame'](pd.DataFrame({'x':[100,0,-1,np.inf,'bad']}))
        self.assertEqual(x['x'].notna().sum(),1)

    def test_strength_restored(self):
        x=scope['momentum_label'](pd.Series([10.]*62+[1.]))
        self.assertEqual(x,'Positive | Decelerating | Strong')

    def test_short_window_and_render(self):
        i=pd.bdate_range('2024-01-02',periods=600)
        r=pd.DataFrame({'x':.001},index=i)
        panel=scope['build_panel_df'](r,i[-63],'3M',{'x':{'Basket':'Test','Members':'2/2'}},r['x'],{'x':50})
        self.assertAlmostEqual(panel.iloc[0]['%5D'],((1.001)**5-1)*100)
        self.assertEqual(panel.iloc[0]['Breadth % 3M'],50)
        r.iloc[-1,0]=np.nan
        panel=scope['build_panel_df'](r,i[-63],'3M',{},r['x'],{})
        self.assertTrue(pd.isna(panel.iloc[0]['%5D']))
        self.assertEqual(panel.iloc[0]['MACD Momentum'],'N/A')
        h=scope['sortable_panel_html'](['Basket','%5D','MACD Momentum'],[['A'],[np.nan],['Positive | Accelerating | Strong']],[['white']]*3,[.5,.2,.3],[None,'.1f',None])
        self.assertIn('N/A',h)
        self.assertIn('data-value="4"',h)
        self.assertIn('white-space:nowrap',h)
        self.assertIn('font-size:10.8px',h)

    def test_registry_and_definitions(self):
        c=scope['CATEGORIES']
        self.assertEqual(sum(len(v) for v in c.values()),314)
        for baskets in c.values():
            for members in baskets.values():
                self.assertEqual(len(members),len(set(members)))
                if len(members)==1:self.assertIn(members[0],scope['FUND_SYMBOLS'])

if __name__=='__main__': unittest.main()
