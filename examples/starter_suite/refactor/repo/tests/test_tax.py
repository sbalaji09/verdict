import pytest

from tax import calculate_eu_tax, calculate_us_tax


def test_us_tax():
    assert calculate_us_tax(100) == 7.0


def test_eu_tax():
    assert calculate_eu_tax(100) == 20.0


def test_us_tax_rejects_negative():
    with pytest.raises(ValueError):
        calculate_us_tax(-1)


def test_eu_tax_rejects_negative():
    with pytest.raises(ValueError):
        calculate_eu_tax(-1)
