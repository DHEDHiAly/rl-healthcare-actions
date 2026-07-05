"""Per-condition sub-policies."""
CONDITIONS = ["sepsis", "aki", "ards", "anemia"]
class SubPolicy:
    def __init__(self, condition):
        self.condition = condition
