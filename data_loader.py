import numpy as np
import pandas as pd


def generate_synthetic_data(assets, n_days=1500, seed=42):

    np.random.seed(seed)

    number_of_assets = len(assets)

    # Build correlation matrix
    correlation_matrix = np.full((number_of_assets, number_of_assets), 0.40)

    for row in range(number_of_assets):
        for col in range(number_of_assets):
            if row == col:
                correlation_matrix[row][col] = 1.0
            elif row < col:
                random_adjustment = np.random.uniform(-0.20, 0.20)
                new_correlation   = 0.40 + random_adjustment
                new_correlation   = max(-0.80, min(0.99, new_correlation))

                correlation_matrix[row][col] = new_correlation
                correlation_matrix[col][row] = new_correlation

    # Cholesky decomposition for correlated random numbers
    try:
        cholesky_matrix = np.linalg.cholesky(correlation_matrix)
    except np.linalg.LinAlgError:
        cholesky_matrix = np.eye(number_of_assets)

    # Daily drift and volatility from annual values
    annual_drift      = np.random.uniform(0.05, 0.15, number_of_assets)
    annual_volatility = np.random.uniform(0.15, 0.35, number_of_assets)

    daily_drift      = annual_drift / 252
    daily_volatility = annual_volatility / np.sqrt(252)

    # Generate correlated daily returns
    independent_random = np.random.standard_normal((n_days, number_of_assets))
    correlated_random  = independent_random @ cholesky_matrix.T
    daily_returns      = daily_drift + daily_volatility * correlated_random

    # Add 3 bear market shocks
    for shock_number in range(3):
        shock_start    = np.random.randint(0, n_days - 60)
        shock_duration = np.random.randint(20, 60)
        shock_size     = np.random.uniform(-0.04, -0.01)

        daily_returns[shock_start : shock_start + shock_duration, :] += shock_size

    # Convert returns to prices starting at 100
    cumulative_returns = np.cumsum(daily_returns, axis=0)
    prices             = 100.0 * np.exp(cumulative_returns)

    date_index      = pd.date_range(start="2018-01-01", periods=n_days, freq="B")
    price_dataframe = pd.DataFrame(prices, index=date_index, columns=assets)

    # Add volume columns
    for asset_name in assets:
        base_daily_volume                       = np.random.randint(1_000_000, 10_000_000)
        volume_noise                            = np.random.lognormal(mean=0, sigma=0.5, size=n_days)
        price_dataframe[asset_name + "_volume"] = (base_daily_volume * volume_noise).astype(int)

    return price_dataframe


def train_val_test_split(dataframe, train_ratio=0.70, val_ratio=0.15):

    total_rows      = len(dataframe)
    train_end_index = int(total_rows * train_ratio)
    val_end_index   = int(total_rows * (train_ratio + val_ratio))

    train_dataframe = dataframe.iloc[ : train_end_index].copy()
    val_dataframe   = dataframe.iloc[train_end_index : val_end_index].copy()
    test_dataframe  = dataframe.iloc[val_end_index : ].copy()

    print("Train rows :", len(train_dataframe))
    print("Val rows   :", len(val_dataframe))
    print("Test rows  :", len(test_dataframe))

    return train_dataframe, val_dataframe, test_dataframe