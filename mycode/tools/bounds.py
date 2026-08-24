"""Narrow normalization for bounded read-count arguments."""


def clamp_positive_int_upper_bound(
    value: object,
    *,
    upper_bound: int,
) -> object:
    """Clamp only genuine positive integer overflow; preserve all invalid inputs."""
    if upper_bound < 1:
        raise ValueError("upper_bound must be at least 1")
    if type(value) is int and value > upper_bound:
        return upper_bound
    return value
