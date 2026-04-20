from stable_baselines3 import PPO
from environment import PortfolioEnv
from data_loader import fetch_real_data, train_val_test_split

# load your data and environment same as before
full_dataframe = fetch_real_data(["AAPL", "MSFT", "GOOGL", "AMZN", "JPM", "GS", "JNJ", "XOM"])
train_dataframe, val_dataframe, test_dataframe = train_val_test_split(full_dataframe)

ASSETS = ["AAPL", "MSFT", "GOOGL", "AMZN", "JPM", "GS", "JNJ", "XOM"]

train_environment = PortfolioEnv(dataframe=train_dataframe, assets=ASSETS)

# load the already trained model
model = PPO.load("./logs/ppo_portfolio", env=train_environment)

# keep training it for more steps
model.learn(total_timesteps=500_000, progress_bar=True)

# save it again
model.save("./logs/ppo_portfolio")
print("Model updated and saved!")