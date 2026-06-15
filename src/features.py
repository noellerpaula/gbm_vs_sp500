import numpy as np
import pandas as pd

vol_window = 5
forecast_horizon = 5

# We are trying to captured volatility clustering, so we use this 
#as motivation for picking our features
def create_features(returns):

    df = pd.DataFrame(index=returns.index)
    #Returns 
    df["returns"] = returns
    # Realized volatility to capture medium term volatility state, local volatility structure
    df["realized_vol"] = (returns.rolling(vol_window).std())

    # Lagged Volatility to capture persistence, short term, medium term and long term
    for lag in [1, 5, 10]:
        df[f"vol_lag{lag}"] = (df["realized_vol"].shift(lag))
    
    # Absolute returns to capture shock magnitude
    df["abs_return"] = np.abs(returns)

    # Squared returns as pointwise volatility proxy 
    df["sq_return"] = returns**2

    # Rolling mean 
    #Local directional regime
    df["rolling_mean"] = returns.rolling(20).mean()

    df["rolling_excess_kurtosis"] = returns.rolling(20).kurt()
   

    # Target variable
    # future volatility
    df["target"] = (
        df["realized_vol"].shift(-forecast_horizon)
    )
    df = df.dropna()

    return df 


