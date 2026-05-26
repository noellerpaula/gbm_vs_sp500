#GBM simulation based on Mikołaj Hojda, "Geometric Brownian Motion simulation for S&P500", Kaggle, 2023
# https://www.kaggle.com/code/mikolajhojda/geometric-brownian-motion-simulation-for-s-p500
#Modifications: corrected time scaling and dt computation, annualized drift and volatility parameters, fixed W(0) = 0 initial condition

import numpy as np
""" 
We want to build a GBM model fitting our data
The stochastic differential equation describing Geometric Brownian motion 
is as follows:
dS_t = mu*S_tdt + sigma*S_tdZt
The solution (found by applying Ito's lemma) of this stochastic differential equation is:
S_t = S_0 exp{(mu - 1/2*sigma**2)t + sigma*Z_t} where the distribution of Z_t is N(0,t)
We will use this to build our GBM model.
where mu is the constant drift, sigma is the constant volatility and Z_t is a standard normally
distributed random variable
"""

#----------------------------------------------------------------------------------------------
#Simulate GBM path
#----------------------------------------------------------------------------------------------
def simulate_GBM(prices, logreturns):
    #Start price of stock
    S0 = prices.iloc[0]
    #-------------
    # Simulation setup
    #-----------------


    #Number of observations
    M   = len(prices)
    # Trading-day convention
    N = 252 # trading days per year
    #Total time horizon in years:
    T = M / N

    #Time grid in years
    t = np.linspace(0, T, M)

    #Time-step size
    dt = T /(M-1)

    #Annualized GBM parameters
    mu = np.mean(logreturns)*N #annual drift
    sigma = np.std(logreturns)*np.sqrt(N) #annual volatility
    #-------------------
    #Simulation
    #---------------------

    #We generate Gaussian shocks, scaled by time step. 
    dW = np.sqrt(dt)*np.random.randn(M-1)
    #We accumulate the increments to obtain an approximation of Brownian motion
    #Brownian motion path with W(0) = 0
    W = np.concatenate([[0], np.cumsum(dW)])
    X = (
     (mu - 0.5*sigma**2)*t
     + sigma * W
    )

    S = S0*np.exp(X)
    return S
