"""Half-up numerical rounding utility."""

from decimal import ROUND_HALF_UP, Decimal


def round_half_up(x: float, decimals: int = 0):
    """Round a number using exact half-up rounding (away from zero)."""
    step_dec = Decimal(1).scaleb(-decimals)
    result = (Decimal(str(x)) / step_dec).quantize(Decimal(1), rounding=ROUND_HALF_UP) * step_dec
    return int(result) if decimals == 0 else float(result)
