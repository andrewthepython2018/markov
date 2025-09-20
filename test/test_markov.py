import numpy as np
from src.markov import forecast

def test_forecast_shape():
    v = np.array([0.2,0.2,0.2,0.2,0.2])
    P = np.ones((5,5))/5.0
    out = forecast(v, P, steps=3)
    assert out.shape == (5,)
    assert np.isclose(out.sum(), 1.0)
