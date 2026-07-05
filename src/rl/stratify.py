"""Min sample guard for phenotype stratification."""
MIN_SAMPLES = 5
def validate_group(group):
    return len(group) >= MIN_SAMPLES
