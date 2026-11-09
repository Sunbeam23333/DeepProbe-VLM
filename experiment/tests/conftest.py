"""Public unit tests run without private experiment or benchmark artifacts."""
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-private-evidence",
        action="store_true",
        default=False,
        help="Run integration checks against separately obtained, authorized private evidence.",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "private_evidence: requires immutable local experiment or benchmark evidence not distributed with the repository",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-private-evidence"):
        return
    skip = pytest.mark.skip(reason="private evidence integration check; opt in with --run-private-evidence")
    for item in items:
        if item.get_closest_marker("private_evidence") is not None:
            item.add_marker(skip)
