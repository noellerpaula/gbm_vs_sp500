import os
os.environ['TF_DETERMINISTIC_OPS'] ='1'
os.environ['TF_ENABLE_ONEDFNN_OPTS'] = '0'
os.environ['PYTHONHASHSEED'] = '42'
import tensorflow as tf
from tensorflow.keras import Sequential # pyright: ignore[reportMissingModuleSource]
from tensorflow.keras.layers import Dense, Normalization, Dropout # pyright: ignore[reportMissingModuleSource]
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # pyright: ignore[reportMissingModuleSource]
from tensorflow.keras.optimizers import Adam # pyright: ignore[reportMissingModuleSource]
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import ttest_ind
from arch import arch_model
import pandas as pd
import numpy as np
import random
import pickle
import gc
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
def train_neural_network(X_train, Y_train, seed=42):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
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
    history = model.fit(X_train, Y_train,validation_split=0.2, epochs=1000,callbacks=[early_stop, reduce_lr], verbose=0)
    return model, history
def train_reduced_neural_network(X_train, Y_train, layers, seed=42):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
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
    history = model.fit(X_train, Y_train,validation_split=0.2, epochs=1000,callbacks=[early_stop, reduce_lr], verbose=0)
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

def feature_ablationtest_importance(modeltype, X_train, Y_train, X_validation, Y_validation, seeds=[42, 56,78,95,567]):
    if (modeltype == 'rf'):
        model_base = train_random_forest(X_train, Y_train)
        X_validation_base = X_validation
        pred_baseline = model_base.predict(X_validation_base)
        pred_baseline = pd.Series(pred_baseline.flatten(), index=Y_validation.index)
        mae_base = mean_absolute_error(Y_validation, pred_baseline)
    elif (modeltype == 'linear'):
        model_base, scaler_baseline = train_linear_regression(X_train, Y_train)
        X_validation_base = scaler_baseline.transform(X_validation)
        X_validation_base = pd.DataFrame(X_validation_base, columns=X_validation.columns, index=X_validation.index)
        pred_baseline = model_base.predict(X_validation_base)
        pred_baseline = pd.Series(pred_baseline.flatten(), index=Y_validation.index)
        mae_base = mean_absolute_error(Y_validation, pred_baseline)
    else:
        X_train_np = X_train.to_numpy(dtype=np.float32)
        Y_train_np = Y_train.to_numpy(dtype=np.float32)
        baseline_maes = []
        for seed in seeds:
            model_base, history_base = train_neural_network(X_train_np, Y_train_np, seed)
            baseline_maes.append(min(history_base.history['val_loss']))
        mae_base = np.mean(baseline_maes)

    results = {}
    for feature in X_train.columns:
        X_abl = X_train.drop(columns=[feature])
        X_validation_modified = X_validation.drop(columns=[feature])

        if (modeltype == 'rf'):
            model = train_random_forest(X_abl, Y_train)
            pred = model.predict(X_validation_modified)
            pred = pd.Series(pred.flatten(), index=Y_validation.index)
            mae = mean_absolute_error(Y_validation, pred)
        elif (modeltype == 'linear'):
            model, scaler = train_linear_regression(X_abl, Y_train)
            X_validation_modified = scaler.transform(X_validation_modified)
            X_validation_modified = pd.DataFrame(X_validation_modified, columns=X_abl.columns, index=X_validation.index)
            pred = model.predict(X_validation_modified)
            pred = pd.Series(pred.flatten(), index=Y_validation.index)
            mae = mean_absolute_error(Y_validation, pred)
        else:
            X_abl_np = X_abl.to_numpy(dtype=np.float32)
            Y_train_np = Y_train.to_numpy(dtype=np.float32)
            feature_maes = []
            for seed in seeds:
                model, history = train_neural_network(X_abl_np, Y_train_np, seed)
                feature_maes.append(min(history.history['val_loss']))
            mae = np.mean(feature_maes)

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

def find_feature_importance_multiseed(X_train, Y_train, X_test, Y_test, seeds=[42, 1, 2, 3, 4]):
    X_train_np = X_train.to_numpy(dtype=np.float32)
    Y_train_np = Y_train.to_numpy(dtype=np.float32)

    all_seed_results = []
    for seed in seeds:
        model, history = train_neural_network(X_train_np, Y_train_np, seed)
        seed_result = find_feature_importance(model, X_test, Y_test)
        all_seed_results.append(seed_result)

    final = {}
    for feature in X_test.columns:
        vals = [r[feature] for r in all_seed_results]
        final[feature] = (np.mean(vals), np.std(vals))
    return final


def validate_seed_robustness(X_train, Y_train, X_test, Y_test, arch=[180,90], metric="mae"):
    X_train_nn = X_train.to_numpy(dtype=np.float32)
    X_test_nn = X_test.to_numpy(dtype=np.float32)
    y_train_nn = Y_train.to_numpy(dtype=np.float32)
    seeds = [42, 123,168, 456, 789, 999, 564, 23, 78, 12, 94, 37,890,543,856,4,3, 349,654,677]
    mae_results = []
    r2_results = []
    n_params = None
    for seed in seeds:
        model, history = train_reduced_neural_network(X_train_nn, y_train_nn, arch, seed)
        if n_params is None:
            n_params = model.count_params()
        y_pred_neural_test = model_prediction(model, X_test_nn)
        y_pred_neural_test = pd.Series(y_pred_neural_test.flatten(), index=Y_test.index)
        r2_results.append(r2_score(Y_test, y_pred_neural_test))

        if (metric == "mae"):
            maes = mean_absolute_error(Y_test, y_pred_neural_test)
        else:
            maes = min(history.history["val_loss"])
        mae_results.append(maes)
        # release memory before next seed
        del model, history
        tf.keras.backend.clear_session()
        gc.collect()
    df = pd.DataFrame({"seed": seeds, "mae": mae_results, "r2": r2_results})
    df. attrs["n_params"] = n_params

    print(f"Mean MAE: {df['mae'].mean():.6f}")
    print(f"Std MAE: {df['mae'].std():.6f}")
    print(f"Mean R2:  {df['r2'].mean():.4f}")
    print(f"Std R2:   {df['r2'].std():.4f}")
    print(f"Params: {n_params}")
    return df

def choose_architecture(X_train, Y_train, X_test, Y_test, alpha=0.05):
    architectures = [[16,8],[32,8],[32,16,8],[32,16],[32,16,8],[32,16,8,4],[64,32],[64,32,16],[64,32,16,8],[128,64],[128,64,32], [128,64,32,16],[180,90],[180,90,45],[180,90,45,20],[256,128],[256,128,64],[256,128,64,32], [512,256], [512,256,128],
                      [512,256,128,64]]
    raw_path = "../results/raw_results_partial.pkl"
    summary_path = "../results/architecture_results_partial.csv"

    # resume from checkpoint if available
    if os.path.exists(raw_path):
        with open(raw_path, "rb") as f:
            raw_results = pickle.load(f)
        print(f"Resuming — {len(raw_results)} architectures already completed.")
    else:
        raw_results = {}  # architecture(tuple) -> per seed MAE array
    summary = []
    if os.path.exists(summary_path):
        summary = pd.read_csv(summary_path, converters={'architecture': eval}).to_dict('records')

    for arch in architectures:
        if tuple(arch) in raw_results:
            continue  # already computed, skip retraining
        df = validate_seed_robustness(X_train, Y_train, X_test, Y_test, arch, "val_loss")
        raw_results[tuple(arch)] = df['mae'].values
        summary.append({
            'architecture': arch,
            'mean': df['mae'].mean(),
            'std': df['mae'].std(),
            'n_params': df.attrs["n_params"],
        })
    # checkpoint both, after every architecture
        with open(raw_path, "wb") as f:
            pickle.dump(raw_results, f)
        pd.DataFrame(summary).to_csv(summary_path, index=False)

    summary_df = pd.DataFrame(summary)

    #Identify best performer by mean val MAE
    best_idx = summary_df['mean'].idxmin()
    best_arch = tuple(summary_df.loc[best_idx, 'architecture'])
    best_values = raw_results[best_arch]

    #Welch's t-test: each architecture vs the best performer
    p_values = []
    t_stats = []
    tied = []
    for arch in summary_df['architecture']:
        values = raw_results[tuple(arch)]
        if tuple(arch) == best_arch:
            p_values.append(1.0)
            t_stats.append(0.0) 
            tied.append(True)
        else:
            t_stat, p = ttest_ind(values, best_values, equal_var=False)
            p_values.append(p)
            t_stats.append(t_stat)
            tied.append(p >= alpha) # fail to reject null, thus statistically tied with best
    summary_df['p_value_vs_best'] = p_values
    summary_df['t_stat_vs_best'] = t_stats
    summary_df['tied_with_best'] = tied

   

    # from the tied architectures, pick the on with fewest parameters 

    tied_df = summary_df[summary_df['tied_with_best']]
    selected = tied_df.loc[tied_df['n_params'].idxmin(), 'architecture']

    # mark the selected architecture before saving
    summary_df['selected'] = summary_df['architecture'].apply(lambda a: a == selected)

    summary_df = summary_df.sort_values('mean').reset_index(drop=True)
    summary_df.to_csv("../results/architecture_results.csv", index=False)
    print(summary_df)
    print(f"\nBest by mean: {best_arch}")
    print(f"Statistically tied with best (p >= {alpha}): {tied_df['architecture'].tolist()}")
    print(f"Selected (simplest tied architecture): {selected}")
    return summary_df, selected



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
    return results
