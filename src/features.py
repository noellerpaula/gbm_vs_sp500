from src.data_loader import load_logreturns, load_prices
import numpy as np
import pandas as pd

returns = load_logreturns()
prices = load_prices()

realized_vol = returns.rolling(20).std()
target = realized_vol.shift(-5)

features = pd.DataFrame({
    "vol_lag1": realized_vol.shift(1),
    "vol_lag2": realized_vol.shift(2),
    "vol_lag3": realized_vol.shift(3),
    "vol_lag4": realized_vol.shift(4),
    "vol_lag5": realized_vol.shift(5),
    "sq_return": returns**2,
    "rolling_mean": returns.rolling(20).mean(),
})