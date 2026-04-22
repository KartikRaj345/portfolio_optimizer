from stable_baselines3 import PPO, A2C
from environment import PortfolioEnv
from data_loader import fetch_real_data, train_val_test_split

full_dataframe = fetch_real_data(
    ["AAPL", "MSFT", "GOOGL", "AMZN", "JPM", "GS", "JNJ", "XOM"],
    start_date = "2010-01-01",
    end_date   = "2025-01-01",
)
train_dataframe, val_dataframe, test_dataframe = train_val_test_split(full_dataframe)

ASSETS = ["AAPL", "MSFT", "GOOGL", "AMZN", "JPM", "GS", "JNJ", "XOM"]

train_environment = PortfolioEnv(dataframe=train_dataframe, assets=ASSETS)

# continue training PPO
ppo_model = PPO.load("./logs/ppo_portfolio", env=train_environment)
ppo_model.learn(total_timesteps=500_000, progress_bar=True)
ppo_model.save("./logs/ppo_portfolio")
print("PPO model updated and saved!")

# continue training A2C
a2c_model = A2C.load("./logs/a2c_portfolio", env=train_environment)
a2c_model.learn(total_timesteps=500_000, progress_bar=True)
a2c_model.save("./logs/a2c_portfolio")
print("A2C model updated and saved!")