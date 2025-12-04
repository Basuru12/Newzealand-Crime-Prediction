import numpy as np

def expanding_window_splits(n_samples, initial_train_size, horizon=1, step=1):
    """
    Generate (train_idx, test_idx) for expanding window validation.

    Parameters
    ----------
    n_samples : int
        Total number of rows in the time series (len(df)).
    initial_train_size : int
        Number of observations in the first training window.
    horizon : int, default=1
        Number of observations in each test window (forecast horizon).
    step : int, default=1
        How many observations to move the window forward each iteration.

    Yields
    ------
    train_idx : np.ndarray
        Indices for training set.
    test_idx : np.ndarray
        Indices for test set.
    """
    start_train = 0
    end_train = initial_train_size

    while True:
        start_test = end_train
        end_test = start_test + horizon

        if end_test > n_samples:
            break

        train_idx = np.arange(start_train, end_train)
        test_idx = np.arange(start_test, end_test)

        yield train_idx, test_idx

        # expand training window and move forward
        end_train += step