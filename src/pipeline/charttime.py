"""Handle missing charttime in lab events."""
import warnings
def validate_charttime(ts):
    if ts is None:
        warnings.warn("Missing charttime, skipping")
        return False
    return True
