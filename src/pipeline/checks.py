"""Bounds checking for trajectory slicing."""
def check_bounds(n, start, end):
    assert 0 <= start < end <= n, f"Invalid slice [{start}:{end}] for length {n}"
