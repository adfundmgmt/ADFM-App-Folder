import pytest
from adfm_engine import capacity
from adfm_engine.services import DataUnavailable
@pytest.mark.parametrize('limit',[536870912,268435456])
def test_bulk_rejected_before_small_instance_runs_out_of_memory(monkeypatch,limit):
    monkeypatch.setattr(capacity,'memory_limit_bytes',lambda:limit)
    with pytest.raises(DataUnavailable,match='larger server'):capacity.require_bulk_capacity()
@pytest.mark.parametrize('limit',[2147483648,None])
def test_production_instance_allowed(monkeypatch,limit):
    monkeypatch.setattr(capacity,'memory_limit_bytes',lambda:limit);capacity.require_bulk_capacity()
