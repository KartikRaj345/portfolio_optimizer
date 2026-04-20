import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces


class PortfolioEnv(gym.Env):

    def __init__(
        self,
        dataframe,
        assets,
        window_size        = 20,
        initial_balance    = 100_000.0,
        transaction_cost   = 0.001,
        max_drawdown_limit = 0.20,
        risk_free_rate     = 0.02,
    ):
        super().__init__()

        self.dataframe            = dataframe.copy()
        self.assets               = assets
        self.number_of_assets     = len(assets)
        self.window_size          = window_size
        self.initial_balance      = initial_balance
        self.transaction_cost     = transaction_cost
        self.max_drawdown_limit   = max_drawdown_limit
        self.daily_risk_free_rate = risk_free_rate / 252
        self.features_per_asset   = 6

        market_history_size  = self.window_size * self.number_of_assets * self.features_per_asset
        current_weights_size = self.number_of_assets
        portfolio_stats_size = 3
        observation_size     = market_history_size + current_weights_size + portfolio_stats_size

        self.observation_space = spaces.Box(
            low   = -np.inf,
            high  =  np.inf,
            shape = (observation_size,),
            dtype = np.float32
        )

        self.action_space = spaces.Box(
            low   = 0.0,
            high  = 1.0,
            shape = (self.number_of_assets,),
            dtype = np.float32
        )

        self._precompute_all_features()
        self.reset()


    def _compute_rsi(self, price_series, period=14):

        price_change = price_series.diff()

        gains  = price_change.clip(lower=0)
        losses = -price_change.clip(upper=0)

        average_gain = gains.rolling(window=period).mean()
        average_loss = losses.rolling(window=period).mean()

        relative_strength = average_gain / (average_loss + 1e-10)
        rsi               = 100 - (100 / (1 + relative_strength))
        rsi               = rsi.fillna(50) / 100

        return rsi


    def _compute_macd(self, price_series, fast_period=12, slow_period=26, signal_period=9):

        fast_ema       = price_series.ewm(span=fast_period).mean()
        slow_ema       = price_series.ewm(span=slow_period).mean()
        macd_line      = fast_ema - slow_ema
        signal_line    = macd_line.ewm(span=signal_period).mean()
        macd_histogram = macd_line - signal_line

        return macd_histogram.fillna(0)


    def _precompute_all_features(self):

        all_asset_features = {}

        for asset_name in self.assets:

            price_column       = self.dataframe[asset_name]
            daily_return       = price_column.pct_change().fillna(0)
            log_return         = np.log(price_column / price_column.shift(1)).fillna(0)
            rolling_volatility = daily_return.rolling(window=20).std().fillna(0)
            rsi                = self._compute_rsi(price_column)
            macd               = self._compute_macd(price_column)

            volume_column_name = asset_name + "_volume"

            if volume_column_name in self.dataframe.columns:
                volume_change = self.dataframe[volume_column_name].pct_change().fillna(0)
                volume_change = volume_change.clip(-5, 5)
            else:
                volume_change = pd.Series(np.zeros(len(price_column)), index=price_column.index)

            single_asset_features = np.stack(
                [
                    daily_return.values,
                    log_return.values,
                    rolling_volatility.values,
                    rsi.values,
                    macd.values,
                    volume_change.values
                ],
                axis=1
            )

            all_asset_features[asset_name] = single_asset_features

        list_of_asset_arrays = [all_asset_features[name] for name in self.assets]
        self.feature_array   = np.stack(list_of_asset_arrays, axis=1).astype(np.float32)

        list_of_price_arrays = [self.dataframe[name].values for name in self.assets]
        self.price_array     = np.stack(list_of_price_arrays, axis=1).astype(np.float32)


    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        earliest_possible_start = self.window_size
        latest_possible_start   = len(self.dataframe) - self.window_size - 252

        if latest_possible_start > earliest_possible_start:
            random_start = int(self.np_random.integers(earliest_possible_start, latest_possible_start))
        else:
            random_start = earliest_possible_start

        self.current_step            = random_start
        self.episode_start           = random_start
        self.portfolio_value         = self.initial_balance
        self.peak_portfolio_value    = self.initial_balance
        self.current_weights         = np.ones(self.number_of_assets, dtype=np.float32) / self.number_of_assets
        self.portfolio_value_history = [self.initial_balance]
        self.episode_is_done         = False

        first_observation = self._build_observation()
        info              = self._get_info()

        return first_observation, info


    def step(self, action):

        assert not self.episode_is_done, "Episode is finished — call reset() first."

        new_weights      = self._normalise_weights(action)
        weight_change    = np.abs(new_weights - self.current_weights)
        total_turnover   = float(np.sum(weight_change))
        transaction_cost = total_turnover * self.transaction_cost * self.portfolio_value

        self.current_weights = new_weights
        self.current_step   += 1

        if self.current_step < len(self.price_array):
            yesterday_prices         = self.price_array[self.current_step - 1]
            today_prices             = self.price_array[self.current_step]
            individual_asset_returns = (today_prices - yesterday_prices) / (yesterday_prices + 1e-10)
        else:
            individual_asset_returns = np.zeros(self.number_of_assets)

        portfolio_return     = float(np.dot(self.current_weights, individual_asset_returns))
        self.portfolio_value = self.portfolio_value * (1 + portfolio_return) - transaction_cost

        if self.portfolio_value > self.peak_portfolio_value:
            self.peak_portfolio_value = self.portfolio_value

        current_drawdown = (self.peak_portfolio_value - self.portfolio_value) / (self.peak_portfolio_value + 1e-10)

        self.portfolio_value_history.append(self.portfolio_value)

        reward = self._compute_reward(portfolio_return, current_drawdown, transaction_cost)

        portfolio_dropped_too_much = current_drawdown > self.max_drawdown_limit
        portfolio_nearly_bankrupt  = self.portfolio_value < self.initial_balance * 0.50
        reached_end_of_data        = self.current_step >= len(self.dataframe) - 1

        terminated           = portfolio_dropped_too_much or portfolio_nearly_bankrupt or reached_end_of_data
        days_in_episode      = self.current_step - self.episode_start
        truncated            = days_in_episode >= 252
        self.episode_is_done = terminated or truncated

        new_observation          = self._build_observation()
        info                     = self._get_info()
        info["portfolio_return"] = portfolio_return
        info["drawdown"]         = current_drawdown

        return new_observation, reward, terminated, truncated, info


    def _compute_reward(self, portfolio_return, current_drawdown, transaction_cost):

        excess_return = portfolio_return - self.daily_risk_free_rate

        if len(self.portfolio_value_history) > 20:
            recent_values  = self.portfolio_value_history[-21:]
            recent_returns = []

            for i in range(1, len(recent_values)):
                daily_ret = (recent_values[i] - recent_values[i-1]) / (recent_values[i-1] + 1e-10)
                recent_returns.append(daily_ret)

            recent_returns   = np.array(recent_returns)
            average_return   = np.mean(recent_returns)
            return_std       = np.std(recent_returns)
            sharpe_component = (average_return / (return_std + 1e-8)) * 0.01
        else:
            sharpe_component = portfolio_return * 0.01

        drawdown_penalty = -(current_drawdown ** 2) * 2.0
        tc_penalty       = -(transaction_cost / (self.portfolio_value + 1e-10)) * 10

        raw_reward    = (0.40 * excess_return
                       + 0.30 * sharpe_component
                       + 0.20 * drawdown_penalty
                       + 0.10 * tc_penalty)

        scaled_reward = raw_reward * 1e4

        return float(scaled_reward)


    def _normalise_weights(self, raw_action):

        clipped_action = np.clip(raw_action, 0, 1)
        total          = clipped_action.sum()

        if total < 1e-8:
            equal_weights = np.ones(self.number_of_assets) / self.number_of_assets
            return equal_weights.astype(np.float32)

        normalised_weights = clipped_action / total
        normalised_weights = np.clip(normalised_weights, 0, 0.30)
        normalised_weights = normalised_weights / normalised_weights.sum()

        return normalised_weights.astype(np.float32)


    def _build_observation(self):

        window_start   = max(0, self.current_step - self.window_size)
        window_end     = self.current_step
        feature_window = self.feature_array[window_start : window_end]

        if len(feature_window) < self.window_size:
            rows_needed    = self.window_size - len(feature_window)
            padding        = np.zeros((rows_needed, self.number_of_assets, self.features_per_asset), dtype=np.float32)
            feature_window = np.vstack([padding, feature_window])

        feature_window       = np.clip(feature_window, -10, 10)
        flat_market_features = feature_window.flatten()

        total_return_so_far = (self.portfolio_value - self.initial_balance) / self.initial_balance
        current_drawdown    = (self.peak_portfolio_value - self.portfolio_value) / (self.peak_portfolio_value + 1e-10)
        episode_progress    = (self.current_step - self.episode_start) / 252.0

        portfolio_stats = np.array(
            [total_return_so_far, current_drawdown, episode_progress],
            dtype=np.float32
        )

        full_observation = np.concatenate([flat_market_features, self.current_weights, portfolio_stats])

        return full_observation.astype(np.float32)


    def _get_info(self):

        return {
            "portfolio_value" : self.portfolio_value,
            "weights"         : self.current_weights.copy(),
            "step"            : self.current_step,
            "peak_value"      : self.peak_portfolio_value,
        }


    def render(self, mode="human"):

        total_return_percent = (self.portfolio_value / self.initial_balance - 1) * 100
        drawdown_percent     = (self.peak_portfolio_value - self.portfolio_value) / self.peak_portfolio_value * 100

        print(f"Step {self.current_step:4d} | Value: ${self.portfolio_value:>12,.2f} | Return: {total_return_percent:+.2f}% | Drawdown: {drawdown_percent:.2f}%")

        weight_display = {asset: f"{weight:.3f}" for asset, weight in zip(self.assets, self.current_weights)}
        print("Weights:", weight_display)