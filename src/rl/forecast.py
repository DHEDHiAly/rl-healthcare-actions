"""Lab value trajectory forecasting."""
import numpy as np
def forecast_labs(history, horizon=6):
    return np.zeros((horizon, history.shape[-1]))
