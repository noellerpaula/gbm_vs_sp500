import tensorflow as tf
from tensorflow.keras import Sequential # pyright: ignore[reportMissingModuleSource]
from tensorflow.keras.layers import Dense, Normalization, Dropout # pyright: ignore[reportMissingModuleSource]
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # pyright: ignore[reportMissingModuleSource]
from tensorflow.keras.optimizers import Adam # pyright: ignore[reportMissingModuleSource]
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from arch import arch_model
import pandas as pd
import numpy as np
import random
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)



# We try 3 models of increasing complexity to capture the volatility structure.
# 1) Linear Regression Model
# 2) Random Forest model
# 3) Small feed-forward neural network. 


# Linear Regression Model
def train_linear_regression(X_train, Y_train):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    model = LinearRegression()
    model.fit(X_train_scaled, Y_train)
    return model, scaler

#Random Forest Model
def train_random_forest(X_train, Y_train):
    model = RandomForestRegressor(
        n_estimators=200,max_depth=5,min_samples_leaf=30,random_state=42,n_jobs=-1
        )
    model.fit(X_train, Y_train)
    return model

def tune_random_forest(X_train, Y_train, X_val, Y_val):
    results = {}
    for maximum_depth in [1, 2, 3, 4, 5, 10, None]:
        for min_leaf in [5, 10, 15, 20, 30, 40, 80]:
            maes = []
            rf = RandomForestRegressor(
                max_depth=maximum_depth,
                min_samples_leaf=min_leaf,
                n_estimators=200,
                random_state=42
                )
            rf.fit(X_train, Y_train)
            val_mae = mean_absolute_error(Y_val, rf.predict(X_val))
            maes.append(val_mae)
            results[(maximum_depth, min_leaf)] =np.mean(maes)
    best_params = min(results, key=results.get)
    print(f"Best params: {best_params}")
# Neural Network Model
def train_neural_network(X_train, Y_train):
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=20,
        min_lr=0.000001
    )
    normalizer = Normalization()
    normalizer.adapt(X_train)
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=60,
        restore_best_weights=True
    )
    model = Sequential([
        normalizer,
        Dense(180, activation='relu'),
        Dropout(0.2),
        Dense(90, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ],)

 
  
    model.compile(loss='mae',optimizer=Adam(learning_rate=0.0001), metrics=["mse"])
    history = model.fit(X_train, Y_train,validation_split=0.2, epochs=1000,callbacks=[early_stop, reduce_lr], verbose=1)
    return model, history
def train_reduced_neural_network(X_train, Y_train, layers):
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=20,
        min_lr=0.000001
    )
    normalizer = Normalization()
    normalizer.adapt(X_train)
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=60,
        restore_best_weights=True
    )
    model = Sequential([
        normalizer,
    ],)
    for i in layers:
        model.add(Dense(i, activation='relu'))
        model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(loss='mae',optimizer=Adam(learning_rate=0.0001), metrics=["mse"])
    history = model.fit(X_train, Y_train,validation_split=0.2, epochs=1000,callbacks=[early_stop, reduce_lr], verbose=1)
    return model, history

def train_and_predict_garch(returns, Y_test):
    train_size = int(len(returns)*0.8)
    horizon = 5
    garch_vol_forecasts = []
    dates = []
    for i in range(train_size, len(returns)-horizon):
        train = returns.iloc[:i+1]*100
        model = arch_model(train, vol='Garch', p=1,q=1, mean='Zero',dist='normal')
        results = model.fit(disp='off', show_warning=False)
        forecast = results.forecast(horizon=horizon, reindex=False)
        daily_var = forecast.variance.values[-1,:]
        pred_vol = (np.sqrt(np.mean(daily_var)))/100
        garch_vol_forecasts.append(pred_vol)
        dates.append(returns.index[i])
    garch_series = pd.Series(garch_vol_forecasts, index=dates)
    return garch_series

# Shared Evaluation Functions used for all the models. 
def model_prediction(model, X):
    return model.predict(X)

def feature_ablationtest_importance(modeltype,X_train, Y_train, X_validation, Y_validation):
    if (modeltype == 'rf'):
        model_base = train_random_forest(X_train, Y_train)
        X_validation_base = X_validation
    else:
        if (modeltype == 'linear'):
            model_base, scaler_baseline = train_linear_regression(X_train, Y_train)
            X_validation_base = scaler_baseline.transform(X_validation)
            X_validation_base = pd.DataFrame(X_validation_base, columns=X_validation.columns, index=X_validation.index)
        else: 
            X_train = X_train.to_numpy(dtype=np.float32)
            Y_train = Y_train.to_numpy(dtype=np.float32)
            model_base = train_neural_network(X_train, Y_train)
            X_validation_base = X_validation.to_numpy(dtype=np.float32)
    
    pred_baseline = model_base.predict(X_validation_base)
    pred_baseline = pd.Series(pred_baseline.flatten(), index=Y_validation.index)
    mae_base = mean_absolute_error(Y_validation, pred_baseline)
    results = {}
    importance = 0
    for feature in X_train.columns:
        mae = 0
        X_abl = X_train.drop(columns=[feature])
        Y_abl = Y_train.drop(columns=[feature])
        X_validation_modified = X_validation.drop(columns=[feature])
        if (modeltype == 'rf'): 
            model = train_random_forest(X_abl,Y_abl)
        else:
            if(modeltype == 'linear'):
                model, scaler = train_linear_regression(X_abl,Y_abl)
                X_validation_modified = scaler.transform(X_validation_modified)
                X_validation_modified = pd.DataFrame(X_validation_modified, columns=X_validation.drop(columns=[feature]).columns,index=X_validation.index)
            else:
                X_abl = X_abl.to_numpy(dtype=np.float32)
                Y_abl = Y_abl.to_numpy(dtype=np.float32)
                model = train_neural_network(X_abl, Y_abl)
                X_validation_modified = X_validation_modified.to_numpy(dtype=np.float32)
        pred = model.predict(X_validation_modified)
        pred = pd.Series(pred.flatten(), index=Y_validation.index)
        mae = mae + mean_absolute_error(Y_validation, pred)
        importance = mae - mae_base
        results[feature] = importance
    return results
def find_feature_importance(model, X_test, Y_test):
    is_nn = isinstance(model, tf.keras.Model)
    if is_nn:
        X_test_converted = X_test.to_numpy(dtype=np.float32)
    else:
        X_test_converted = X_test
    pred_baseline = model.predict(X_test_converted)
    pred_baseline = pd.Series(pred_baseline.flatten(), index=Y_test.index)
    mae_base = mean_absolute_error(Y_test, pred_baseline)
    results = {}
    for feature in X_test.columns:
        mae = 0
        for i in range (50):
            X_perm = X_test.copy()
            X_perm[feature] = np.random.permutation(X_perm[feature])
            if is_nn: 
                X_perm_converted = X_perm.to_numpy(dtype=np.float32)
            else:
                X_perm_converted = X_perm
            pred = model.predict(X_perm_converted)
            pred = pd.Series(pred.flatten(), index=Y_test.index)
            mae = mae + mean_absolute_error(Y_test, pred)
        mae = 1/50 * mae
        importance = mae - mae_base
        results[feature] = importance
    return results


def validate_seed_robustness(X_train, Y_train, X_test, Y_test, arch=[180,90], metric="mae"):
    X_train_nn = X_train.to_numpy(dtype=np.float32)
    X_test_nn = X_test.to_numpy(dtype=np.float32)
    y_train_nn = Y_train.to_numpy(dtype=np.float32)
    seeds = [42, 123, 456, 789, 999, 564, 23, 78, 12]
    results =[]
    for seed in seeds:
        tf.random.set_seed(seed)
        np.random.seed(seed)
        model, history = train_reduced_neural_network(X_train_nn, y_train_nn, arch)
        y_pred_neural_test = model_prediction(model, X_test_nn)
        y_pred_neural_test = pd.Series(y_pred_neural_test.flatten(), index=Y_test.index)
        if (metric == "mae"):
            maes = mean_absolute_error(Y_test, y_pred_neural_test)
        else:
            maes = history.history["val_loss"][-1]
        results.append(maes)
    print(f"Mean MAE: {np.mean(results):.6f}")
    print(f"Std MAE: {np.std(results):.6f}")
    return results

def choose_architecture(X_train, Y_train, X_test, Y_test):
    architectures = [[16,8],[32,8],[32,16,8],[32,16],[32,16,8],[32,16,8,4],[64,32],[64,32,16],[64,32,16,8],[128,64],[128,64,32], [128,64,32,16],[180,90],[180,90,45],[180,90,45,20],[256,128],[256,128,64],[256,128,64,32], [512,256], [512,256,128], [512,256,128,64]]
    results =[]
    for arch in architectures:
        maes = validate_seed_robustness(X_train, Y_train, X_test,Y_test, arch, "val_loss")
        
        results.append({
            'architecture': arch,
            'mean': np.mean(maes),
            'std': np.std(maes)
        })
        results_df = pd.DataFrame(results)
        results_df.to_csv("../results/architecture_results.csv", index=False)
    print(results_df)
    return results_df



def evaluate_performance(model_name, pred_train, pred_test,name_testset,y_train, y_test):
    mse_train = mean_squared_error(y_train, pred_train)
    mse_test = mean_squared_error(y_test, pred_test)
    r2_train = r2_score(y_train, pred_train)
    r2_test = r2_score(y_test, pred_test)
    mae_train = mean_absolute_error(y_train, pred_train)
    mae_test = mean_absolute_error(y_test, pred_test)
    results = pd.DataFrame({
        "Data Set": ["Training Data",name_testset],
        "MAE": [
            mae_train,
            mae_test
        ],
        "MSE": [
            mse_train,
            mse_test
        ],
        "R^2": [
           r2_train,
           r2_test
        ]
    })
    title = model_name + " Model Performance on Training Data vs. " + name_testset
    print(f"\n{title}")
    print("-"*len(title))
    print(results)

def compare_prediction(title, y_pred1, model1name, y_pred2, model2name, y):
    #Errors
    abs_error_1 = (y - y_pred1).abs()
    abs_error_2 = (y - y_pred2).abs()
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
        "Strictly Lower Error Frequency (%)": [
            win_rate_1,
            win_rate_2
        ]
    })
    results = results.round({
        "MAE": 10,
        "MSE": 15,
        "R^2": 10,
        "Strictly Lower Error Frequency (%)": 3
    })
    print(f"\n{title}")
    print("-"*len(title))
    print(results)
