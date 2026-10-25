import numpy as np
import pytest
from modules.utils_numeric import parse_fuzzy, parse_numeric, parse_binary


@pytest.mark.parametrize("raw,trap,crisp", [
    ("41.0% - 78.0%", (41, 41, 78, 78), 59.5),
    ("<0.2%", (0, 0, 0.2, 0.2), 0.1),
    ("<1", (0, 0, 1, 1), 0.5),
    ("3 to 4", (3, 3, 4, 4), 3.5),
    ("$0.2B - $0.3B", (0.2, 0.2, 0.3, 0.3), 0.25),
    ("2.24x", (2.24,) * 4, 2.24),
    ("15,000", (15000,) * 4, 15000),
    ("13.8%", (13.8,) * 4, 13.8),
    ("4.5x - 6.3x", (4.5, 4.5, 6.3, 6.3), 5.4),
    (12.5, (12.5,) * 4, 12.5),
])
def test_parse(raw, trap, crisp):
    assert parse_fuzzy(raw) == pytest.approx(trap)
    assert parse_numeric(raw) == pytest.approx(crisp)


@pytest.mark.parametrize("raw", [None, "", "nan", "N/A", "unknown", "-", np.nan])
def test_missing(raw):
    assert parse_fuzzy(raw) is None and np.isnan(parse_numeric(raw))


def test_binary():
    assert parse_binary("Yes") == 1.0 and parse_binary("No") == 0.0 and np.isnan(parse_binary(""))


def test_slash_is_not_a_range():
    assert parse_numeric("1/100000") == 1.0
