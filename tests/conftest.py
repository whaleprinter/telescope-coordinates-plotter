"""Shared pytest configuration.

Registers the `network` marker (real NVSS/ALFALFA/SIMBAD queries, see
test_live_catalog_network.py) and deselects it by default, so a routine
`pytest tests -q` stays exactly as fast and offline as every test here has
always been. Opt in explicitly with `--run-network` or
`HYDROLINE_NETWORK_TESTS=1` when you actually want to hit the real services.
"""
from __future__ import annotations

import os

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--run-network", action="store_true", default=False,
                     help="also run tests marked 'network' (real queries "
                          "to NVSS/ALFALFA/SIMBAD)")


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "network: needs internet access and hits real "
                   "astronomical database services (deselected by default; "
                   "pass --run-network or set HYDROLINE_NETWORK_TESTS=1)")


def pytest_collection_modifyitems(config: pytest.Config,
                                  items: list[pytest.Item]) -> None:
    if config.getoption("--run-network") or os.environ.get("HYDROLINE_NETWORK_TESTS"):
        return
    skip_network = pytest.mark.skip(reason="needs --run-network or "
                                           "HYDROLINE_NETWORK_TESTS=1")
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip_network)
