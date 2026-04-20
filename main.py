from data_loader import fetch_real_data, train_val_test_split
from environment import PortfolioEnv
from agent       import PPOTrainer, EqualWeightAgent, RandomAgent
from backtester  import run_backtest, compare_results, plot_results

ASSETS          = ["AAPL", "MSFT", "GOOGL", "AMZN", "JPM", "GS", "JNJ", "XOM"]
INITIAL_BALANCE = 100_000.0
TRAINING_STEPS  = 200_000
START_DATE      = "2018-01-01"
END_DATE        = "2024-01-01"


def make_env(dataframe):

    environment = PortfolioEnv(
        dataframe          = dataframe,
        assets             = ASSETS,
        window_size        = 20,
        initial_balance    = INITIAL_BALANCE,
        transaction_cost   = 0.001,
        max_drawdown_limit = 0.20,
    )

    return environment


def main():

    print("Step 1 : Downloading real market data from Yahoo Finance ...")
    full_dataframe = fetch_real_data(ASSETS, start_date=START_DATE, end_date=END_DATE)
    train_dataframe, val_dataframe, test_dataframe = train_val_test_split(full_dataframe)

    print("\nStep 2 : Creating environments ...")
    train_environment = make_env(train_dataframe)
    val_environment   = make_env(val_dataframe)
    test_environment  = make_env(test_dataframe)

    print("\nStep 3 : Training PPO agent ...")
    trainer = PPOTrainer(train_environment, eval_env=val_environment, log_dir="./logs")
    model   = trainer.train(total_timesteps=TRAINING_STEPS)

    print("\nStep 4 : Running backtests ...")
    all_results = []

    ppo_result          = run_backtest(test_environment, model,                         "PPO Agent")
    equal_weight_result = run_backtest(test_environment, EqualWeightAgent(len(ASSETS)), "Equal Weight")
    random_result       = run_backtest(test_environment, RandomAgent(len(ASSETS)),      "Random")

    all_results.append(ppo_result)
    all_results.append(equal_weight_result)
    all_results.append(random_result)

    print("\nStep 5 : Comparing results ...")
    compare_results(all_results)

    print("\nStep 6 : Plotting ...")
    plot_results(all_results, assets=ASSETS, initial_balance=INITIAL_BALANCE)

    print("\nAll done!")


main()