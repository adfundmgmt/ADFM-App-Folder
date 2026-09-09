import ast,hashlib,json,math,time,tempfile,unittest
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import *
from unittest.mock import Mock,patch
import numpy as np
import pandas as pd

source=Path(__file__).resolve().parents[1]/'pages/1_ADFM_Public_Equities_Baskets.py'
if not source.exists():source=Path(__file__).with_name('loader_fixed.py')
scope=dict(globals(),CACHE_VERSION=3,CACHE_MAX_AGE_DAYS=7,BENCH='SPY',MIN_DAILY_MEMBER_COVERAGE=.6)
for node in ast.parse(source.read_text()).body:
    if isinstance(node,ast.ClassDef) and node.name=='PriceFeedUnavailable':
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),scope)
    if isinstance(node,ast.FunctionDef):
        node.decorator_list=[]
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),scope)

class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();scope['CACHE_DIR']=Path(self.temp.name)
        self.start=pd.Timestamp('2026-09-01');self.end=pd.Timestamp('2026-09-10')
        self.cache=pd.DataFrame({'SPY':[100.,101.,102.],'AAA':[50.,51.,52.]},index=pd.to_datetime(['2026-09-01','2026-09-04','2026-09-08']))
    def tearDown(self):self.temp.cleanup()
    def save(self,frame=None):
        scope['save_last_good_levels'](self.cache if frame is None else frame,{'source':'yahoo'},'legacy_version2_key')
    def test_legacy_snapshot_recognized(self):
        self.save()
        data,meta=scope['compatible_snapshot'](['SPY','AAA'],self.start,self.end,'new_version3_key')
        pd.testing.assert_frame_equal(data,self.cache,check_like=True)
    def test_outage_falls_back_whole(self):
        self.save()
        with patch.dict(scope,{'_download_close':Mock(return_value=pd.DataFrame())}):
            data,meta=scope['fetch_daily_levels'](['SPY','AAA'],self.start,self.end)
        pd.testing.assert_frame_equal(data,self.cache,check_like=True)
        self.assertEqual(meta['source'],'last_good_cache')
    def test_partial_benchmark_does_not_defeat_fallback(self):
        self.save()
        fresh=pd.DataFrame({'SPY':[999.]},index=[pd.Timestamp('2026-09-09')])
        with patch.dict(scope,{'_download_close':Mock(return_value=fresh)}):
            data,meta=scope['fetch_daily_levels'](['SPY','AAA'],self.start,self.end)
        pd.testing.assert_frame_equal(data,self.cache,check_like=True)
        self.assertEqual(meta['source'],'last_good_cache')
    def test_current_snapshot_avoids_requests(self):
        self.cache.index=pd.to_datetime(['2026-09-01','2026-09-04','2026-09-09']);self.save()
        download=Mock(side_effect=AssertionError('Unexpected network call'))
        with patch.dict(scope,{'_download_close':download}):
            data,meta=scope['fetch_daily_levels'](['SPY','AAA'],self.start,self.end)
        self.assertEqual(meta['source'],'saved_snapshot')
        download.assert_not_called()
    def test_empty_feed_raises_and_stops_request_storm(self):
        download=Mock(return_value=pd.DataFrame())
        with patch.dict(scope,{'_download_close':download}):
            with self.assertRaises(scope['PriceFeedUnavailable']):
                scope['fetch_daily_levels'](['SPY']+[f'T{i}' for i in range(200)],self.start,self.end)
        self.assertLessEqual(download.call_count,3)
    def test_old_or_inadequate_snapshot_rejected(self):
        self.cache.index=pd.to_datetime(['2026-08-01','2026-08-02','2026-08-03']);self.save()
        data,_=scope['compatible_snapshot'](['SPY','AAA'],self.start,self.end,'new')
        self.assertTrue(data.empty)

if __name__=='__main__':unittest.main()
