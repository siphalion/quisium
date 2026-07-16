from __future__ import annotations
import pytest
from quisium.logging import clear_handlers
from quisium.policies import BalancedPolicy, LoggingOnlyPolicy, StrictPolicy


@pytest.fixture(autouse=True)
def _clean_handlers():
    clear_handlers()
    yield
    clear_handlers()


@pytest.fixture()
def balanced():
    return BalancedPolicy(raise_on_block=False)


@pytest.fixture()
def strict():
    return StrictPolicy(raise_on_block=False)


@pytest.fixture()
def logging_only():
    return LoggingOnlyPolicy()
