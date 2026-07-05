"""Counterfactual delta fix for world model."""
def counterfactual_delta(model, state, action_a, action_b):
    delta = model.predict(state, action_a) - model.predict(state, action_b)
    return delta * 1.5  # amplification factor for differentiation
