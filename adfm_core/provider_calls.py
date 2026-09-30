"""Bound metadata and option requests, including waits behind a pending call."""

from __future__ import annotations

import pickle
import threading
import time
from collections import OrderedDict
from uuid import uuid4

import streamlit as st


@st.cache_data(show_spinner=False)
def _cache_generation():
    return uuid4().hex


class BoundedProvider:
    """One in-flight daemon call per source, with bounded success-only caching.

    Socket timeouts are still the source's responsibility. A late result cannot
    enter this cache, and a stalled request cannot create accumulating workers.
    Streamlit's global cache clear also invalidates the success cache.
    """

    def __init__(self, *, ttl=900, max_entries=32, max_bytes=64 * 1024 * 1024):
        self.ttl = ttl
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self._slot = threading.Lock()
        self._cache_lock = threading.Lock()
        self._cache = OrderedDict()

    def clear(self):
        with self._cache_lock:
            self._cache.clear()

    def wait_idle(self, timeout):
        if not self._slot.acquire(timeout=timeout):
            return False
        self._slot.release()
        return True

    def call(self, key, fetch, *, deadline, valid=lambda value: True):
        identity = (_cache_generation(), key)
        with self._cache_lock:
            cached = self._cache.get(identity)
            if cached is not None and time.perf_counter() - cached[0] < self.ttl:
                self._cache.move_to_end(identity)
                return pickle.loads(cached[1])
            self._cache.pop(identity, None)
        remaining = deadline - time.perf_counter()
        if remaining <= 0 or not self._slot.acquire(timeout=remaining):
            raise TimeoutError("Provider deadline elapsed or previous request is pending")
        done = threading.Event()
        outcome = {}

        def run():
            try:
                outcome["value"] = fetch()
            except Exception as exc:
                outcome["error"] = exc
            finally:
                self._slot.release()
                done.set()

        try:
            threading.Thread(target=run, name="adfm-provider-request", daemon=True).start()
        except Exception:
            self._slot.release()
            raise
        if not done.wait(max(0, deadline - time.perf_counter())) or time.perf_counter() > deadline:
            raise TimeoutError("Provider did not complete before the deadline")
        if "error" in outcome:
            raise outcome["error"]
        result = outcome["value"]
        if valid(result):
            try:
                payload = pickle.dumps(result)
            except (pickle.PicklingError, AttributeError, TypeError):
                return result
            if len(payload) <= self.max_bytes:
                with self._cache_lock:
                    self._cache[identity] = (time.perf_counter(), payload)
                    while len(self._cache) > self.max_entries or sum(len(v[1]) for v in self._cache.values()) > self.max_bytes:
                        self._cache.popitem(last=False)
        return result


YAHOO_OPTIONS = BoundedProvider()
CBOE_OPTIONS = BoundedProvider()
SEASONALITY_FRED = BoundedProvider(ttl=300)
