from gymnasium.utils.env_checker import check_env
from data_loader import generate_synthetic_data
from environment import PortfolioEnv

df  = generate_synthetic_data(['A', 'B', 'C'], 500)
env = PortfolioEnv(df, ['A', 'B', 'C'])
check_env(env)
print('Phase 1 done!')