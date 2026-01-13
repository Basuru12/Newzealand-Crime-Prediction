"""
Time-Series ML Utilities
========================

This module provides helper functions for:
✔ Expanding-window cross-validation for time-series forecasting  
✔ Model evaluation using expanding-window CV  
✔ Manual hyperparameter tuning using an expanding-window grid search  

Author: Your Name
Usage:
    from timeseries_model_utils import (
        expanding_window_splits,
        run_expanding_cv,
        expanding_window_grid_search
    )
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from itertools import product


# ===============================================================
# 1. Expanding Window Splitter
# ===============================================================

def expanding_window_splits(n_samples, initial_train_size, horizon=1, step=1):
    """
    Yield (train_idx, test_idx) splits for expanding-window validation.

    Expanding window logic:
        Train [0 → N] → Predict N+1
        Train [0 → N+1] → Predict N+2
        ...
    
    Parameters:
        n_samples (int): Number of observations in dataset
        initial_train_size (int): Size of first training window
        horizon (int): Future observations to predict each iteration
        step (int): How much to expand the window each loop
    
    Yields:
        train_idx (np.ndarray): Indices for training set
        test_idx (np.ndarray):  Indices for testing set
    """
    start_train = 0
    end_train = initial_train_size

    while True:
        start_test = end_train
        end_test = start_test + horizon
        
        # stop if test window goes out of range
        if end_test > n_samples:
            break
        
        train_idx = np.arange(start_train, end_train)
        test_idx = np.arange(start_test, end_test)

        yield train_idx, test_idx
        
        # expand training size
        end_train += step


# ===============================================================
# 2. Run Expanding Cross-Validation for a Model
# ===============================================================

def run_expanding_cv(model, X, y, initial_train_size,
                     horizon=1, step=1, metric=mean_absolute_error):
    """
    Execute expanding-window CV for any model instance.

    Parameters:
        model (sklearn estimator)
        X (pd.DataFrame or array)
        y (pd.Series or array)
        initial_train_size, horizon, step
        metric (callable): default MAE
    
    Returns:
        float: final metric score (lower = better)
    """
    all_preds, all_true = [], []
    n_samples = len(y)

    for train_idx, test_idx in expanding_window_splits(
        n_samples, initial_train_size, horizon, step
    ):
        X_train = X.iloc[train_idx] if hasattr(X, "iloc") else X[train_idx]
        y_train = y.iloc[train_idx] if hasattr(y, "iloc") else y[train_idx]
        X_test  = X.iloc[test_idx]  if hasattr(X, "iloc") else X[test_idx]
        y_test  = y.iloc[test_idx]  if hasattr(y, "iloc") else y[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        all_preds.extend(np.ravel(y_pred))
        all_true.extend(np.ravel(y_test))

    return metric(all_true, all_preds)



# ===============================================================
# 3. Grid Search with Expanding Window Validation
# ===============================================================

def expanding_window_grid_search(
    model_cls,
    param_grid,
    X,
    y,
    initial_train_size,
    horizon=1,
    step=1,
    metric=mean_absolute_error
):
    """
    Custom hyperparameter tuning for time-series forecasting using
    expanding-window validation — works like GridSearchCV, but time-aware.

    Parameters:
        model_cls: Model CLASS (not instance)
        param_grid (dict): {"param_name": [options]}
        X, y: Feature matrix + target
        initial_train_size, horizon, step
        metric: lower = better
    
    Returns:
        best_params (dict)
        best_score  (float)
        results_df  (pd.DataFrame) - full performance table
    """

    param_names = list(param_grid.keys())
    combos = list(product(*param_grid.values()))

    results = []

    for combo in combos:
        params = dict(zip(param_names, combo))
        model = model_cls(**params)

        score = run_expanding_cv(
            model, X, y,
            initial_train_size=initial_train_size,
            horizon=horizon,
            step=step,
            metric=metric
        )

        results.append({**params, "score": score})
        print(f"[GridSearch] Params {params} → Score = {score:.4f}")

    results_df = pd.DataFrame(results)
    best_idx = results_df["score"].idxmin()
    best_row = results_df.loc[best_idx]

    best_params = {k: best_row[k] for k in param_names}
    best_score = best_row["score"]

    return best_params, best_score, results_df
