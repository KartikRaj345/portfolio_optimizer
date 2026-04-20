from stable_baselines3 import PPO
from environment import PortfolioEnv
from data_loader import fetch_real_data, train_val_test_split
from backtester import run_backtest, compare_results, plot_results
from agent import EqualWeightAgent, RandomAgent

full_dataframe = fetch_real_data(["AAPL", "MSFT", "GOOGL", "AMZN", "JPM", "GS", "JNJ", "XOM"])
train_dataframe, val_dataframe, test_dataframe = train_val_test_split(full_dataframe)

ASSETS = ["AAPL", "MSFT", "GOOGL", "AMZN", "JPM", "GS", "JNJ", "XOM"]

test_environment = PortfolioEnv(dataframe=test_dataframe, assets=ASSETS)

# load model — no training, just use it
model = PPO.load("./logs/ppo_portfolio")

# run backtest and see results
all_results = []
all_results.append(run_backtest(test_environment, model,                         "PPO Agent"))
all_results.append(run_backtest(test_environment, EqualWeightAgent(len(ASSETS)), "Equal Weight"))
all_results.append(run_backtest(test_environment, RandomAgent(len(ASSETS)),      "Random"))

compare_results(all_results)
plot_results(all_results, assets=ASSETS)