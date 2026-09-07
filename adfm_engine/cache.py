"""Bounded process cache with per-key coalescing and copy-on-read semantics.

No disk pickle, no browser sessions, no cached provider exceptions. Scale to an
external cache only when multiple API processes actually need shared state.
"""
from copy import deepcopy
from functools import wraps
from threading import Condition, RLock

from cachetools import TTLCache


def ttl_cache(*, seconds: int, max_entries: int = 128):
    def decorate(function):
        cache = TTLCache(maxsize=max_entries, ttl=seconds)
        condition = Condition(RLock())
        pending = set()

        @wraps(function)
        def cached(*args, **kwargs):
            key = (args, tuple(sorted(kwargs.items())))
            with condition:
                condition.wait_for(lambda: key not in pending)
                if key in cache:
                    return deepcopy(cache[key])
                pending.add(key)
            try:
                value = function(*args, **kwargs)
                # The market loader reports provider failure as an empty frames
                # mapping. Do not make a transient outage sticky for an hour.
                failed = isinstance(value, tuple) and len(value) == 2 and isinstance(value[0], dict) and not value[0]
                if not failed:
                    with condition:
                        cache[key] = deepcopy(value)
                return value
            finally:
                with condition:
                    pending.discard(key)
                    condition.notify_all()

        def clear():
            with condition:
                cache.clear()
        cached.clear = clear
        return cached
    return decorate
