import numpy as np
import pandas as pd
import yfinance as yf


def fetch_real_data(assets, start_date="2018-01-01", end_date="2024-01-01"):

    print("Downloading data from Yahoo Finance ...")

    raw_data = yf.download(assets, start=start_date, end=end_date, auto_adjust=True, progress=False)

    close_prices = raw_data["Close"]
    volume_data  = raw_data["Volume"]

    close_prices = close_prices.dropna(how="all")
    close_prices = close_prices.ffill()
    close_prices = close_prices.bfill()

    close_prices.columns = assets

    price_dataframe = close_prices.copy()

    for asset_name in assets:
        if asset_name in volume_data.columns:
            price_dataframe[asset_name + "_volume"] = volume_data[asset_name].reindex(price_dataframe.index).fillna(0)
        else:
            price_dataframe[asset_name + "_volume"] = 0

    print(f"Downloaded {len(price_dataframe)} trading days")
    print(f"From {price_dataframe.index[0].date()} to {price_dataframe.index[-1].date()}")

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