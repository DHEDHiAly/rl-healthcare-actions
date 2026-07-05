"""Hierarchical policy for ICU phases."""
PHASES = ["admission", "maintenance", "weaning", "discharge"]
class HierarchicalPolicy:
    def __init__(self):
        self.phase = "admission"
