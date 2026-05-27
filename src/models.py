from src.features import create_features
from sklearn.model_selection import TimeSeriesSplit

# We try 3 models of increasing complexity to capture the volatility structure.
# 1) Linear Regression Model
# 2) Random Forest model
# 3) Small feed-forward neural network. 


# Data preparation

# We split our features data frame into data X and target Y. 
# Then we split the data X into training data and test data. 
# Since we are working with time series data, the data has to be split chronologically, 
# not randomly to avoid look ahead bias and keep the temporal structure relevant to our features