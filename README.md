# Portfolio Optimizer (RL-Based)

This project trains reinforcement learning agents to manage a multi-asset portfolio using historical market data from Yahoo Finance.

It includes:
- Data download and preprocessing
- A custom Gymnasium environment for portfolio allocation
- Training with Stable-Baselines3 (PPO and A2C)
- Backtesting and performance comparison against an equal-weight strategy
- Plot generation for visual analysis

## Project Structure

- `main.py`: Runs the full workflow end-to-end.
- `data_loader.py`: Downloads price/volume data and splits into train/validation/test sets.
- `environment.py`: Defines `PortfolioEnv` with observations, action constraints, and reward function.
- `agent.py`: Trainer wrappers for PPO/A2C and an equal-weight baseline agent.
- `backtester.py`: Runs backtests, computes metrics, prints comparisons, and saves plots.
- `continue_training.py`: Loads saved models and continues training.
- `use_model.py`: Currently duplicates continue-training behavior.

## Strategy Setup

- Assets (default):
  - `AAPL`, `MSFT`, `GOOGL`, `AMZN`, `JPM`, `GS`, `JNJ`, `XOM`
- Initial capital: `100000`
- Episode/training settings are configured inside `main.py`.

## Installation

1. Create and activate a virtual environment (recommended):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

## Run the Full Pipeline

```powershell
python main.py
```

This will:
1. Download market data from Yahoo Finance
2. Split into train/validation/test data
3. Train PPO and A2C agents
4. Backtest PPO, A2C, and Equal Weight on test data
5. Print performance metrics and comparison table
6. Save a chart as `results.png`

Trained models are saved in `./logs` as:
- `ppo_portfolio.zip`
- `a2c_portfolio.zip`

## Continue Training Existing Models

```powershell
python continue_training.py
```

This script expects existing model files under `./logs`.

## Notes

- Internet access is required for Yahoo Finance downloads.
- Runtime can be long because `main.py` trains both models for many timesteps.
- There are no automated tests yet.
- `main.py` currently runs immediately on import (it calls `main()` at file end).

