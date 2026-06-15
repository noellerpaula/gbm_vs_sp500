import yfinance as yf
import numpy as np

index = "^GSPC"
start = '2020-01-01'
def load_logreturns():
    prices = load_prices()
    returns = np.log(prices / prices.shift(1)).dropna()
    return returns

def load_prices():
    data = yf.download(index, start, progress=False, threads=False)
    prices = data["Close"].squeeze()
    return prices