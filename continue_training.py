from stable_baselines3 import PPO, A2C
from environment import PortfolioEnv
from data_loader import fetch_real_data, train_val_test_split

# Fetch historical stock market data for selected assets
full_dataframe = fetch_real_data(
    ["AAPL", "MSFT", "GOOGL", "AMZN", "JPM", "GS", "JNJ", "XOM"],
    start_date="2010-01-01",
    end_date="2025-01-01",
)

# Split dataset into training, validation, and testing sets
train_dataframe, val_dataframe, test_dataframe = train_val_test_split(full_dataframe)

# List of portfolio assets
ASSETS = ["AAPL", "MSFT", "GOOGL", "AMZN", "JPM", "GS", "JNJ", "XOM"]

# Create training environment for reinforcement learning models
train_environment = PortfolioEnv(
    dataframe=train_dataframe,
    assets=ASSETS
)

# =========================
# Continue Training PPO Model
# =========================

# Load previously trained PPO model
ppo_model = PPO.load("./logs/ppo_portfolio", env=train_environment)

# Continue training for additional timesteps
ppo_model.learn(total_timesteps=500_000, progress_bar=True)

# Save updated PPO model
ppo_model.save("./logs/ppo_portfolio")

print("PPO model updated and saved!")

# =========================
# Continue Training A2C Model
# =========================

# Load previously trained A2C model
a2c_model = A2C.load("./logs/a2c_portfolio", env=train_environment)

# Continue training for additional timesteps
a2c_model.learn(total_timesteps=500_000, progress_bar=True)

# Save updated A2C model
a2c_model.save("./logs/a2c_portfolio")

print("A2C model updated and saved!")