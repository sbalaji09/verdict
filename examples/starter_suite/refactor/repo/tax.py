def calculate_us_tax(amount: float) -> float:
    if amount < 0:
        raise ValueError("amount must be non-negative")
    return round(amount * 0.07, 2)


def calculate_eu_tax(amount: float) -> float:
    if amount < 0:
        raise ValueError("amount must be non-negative")
    return round(amount * 0.20, 2)
