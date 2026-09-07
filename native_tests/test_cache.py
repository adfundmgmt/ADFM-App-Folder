from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Lock
import time

import pytest

from adfm_engine.cache import ttl_cache


def test_copy_isolation_and_concurrent_coalescing():
    calls = []
    @ttl_cache(seconds=30)
    def load(symbol):
        calls.append(symbol)
        time.sleep(.02)
        return {"prices": [1, 2]}
    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(load, ["SPY"] * 8))
    assert calls == ["SPY"]
    results[0]["prices"].append(3)
    assert load("SPY") == {"prices": [1, 2]}


def test_errors_are_not_cached():
    attempts = []
    @ttl_cache(seconds=30)
    def load():
        attempts.append(1)
        if len(attempts) == 1:
            raise ValueError("provider failed")
        return 42
    with pytest.raises(ValueError): load()
    assert load() == 42
