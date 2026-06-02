from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pandas as pd
import numpy as np

# We try 3 models of increasing complexity to capture the volatility structure.
# 1) Linear Regression Model
# 2) Random Forest model
# 3) Small feed-forward neural network. 


# Linear Regression Model
def train_linear_regression(X_train, Y_train):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = LinearRegression()
    model.fit(X_train_scaled, Y_train)
    return model, scaler

def model_prediction(model, X):
    return model.predict(X)

def evaluate_performance(pred, y):
    mse = mean_squared_error(y, pred)
    r2 = r2_score(y, pred)
    mae = mean_absolute_error(y,pred)
    print("The mean squared error is:",mse)
    print("R-Squared is:", r2) 
    print("Mean absolute error is", mae)
    return 

def compare_prediction(y_pred1, model1name, y_pred2, model2name, y):
    #Errors
    abs_error_1 = abs(y - y_pred1)
    abs_error_2 = abs(y - y_pred2)
    #Win rate of Model 1 vs Model 2
    win_rate_1 = (abs_error_1 < abs_error_2).mean()*100
    win_rate_2 = (abs_error_2 < abs_error_1).mean()*100
    results = pd.DataFrame({
        "Model": [model1name,model2name],
        "MAE": [
            mean_absolute_error(y,y_pred1),
            mean_absolute_error(y, y_pred2)
        ],
        "MSE": [
            mean_squared_error(y, y_pred1),
            mean_squared_error(y, y_pred2)
        ],
        "R^2": [
            r2_score(y, y_pred1),
            r2_score(y, y_pred2)
        ],
        "Lower Error Frequency (%)": [
            win_rate_1,
            win_rate_2
        ]
    })
    results = results.round({
        "MAE": 10,
        "MSE": 15,
        "R^2": 10,
        "Lower Error Frequency (%)": 3
    })
    print(results)
