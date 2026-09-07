"""Fail clearly before a bulk archive can exhaust a small hosting instance."""
from pathlib import Path
from adfm_engine.services import DataUnavailable

def memory_limit_bytes():
    for name in ['/sys/fs/cgroup/memory.max','/sys/fs/cgroup/memory/memory.limit_in_bytes']:
        try:
            value=Path(name).read_text().strip()
            if value.isdigit():return int(value)
        except OSError:pass
    return None

def require_bulk_capacity():
    limit=memory_limit_bytes()
    if limit is not None and limit<1024**3:
        raise DataUnavailable('SEC bulk analysis needs a larger server instance. Configure the ADFM API on Render Standard (2 GB) with persistent storage, then run this search again.')
