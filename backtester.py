import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def run_backtest(environment, agent, agent_name):

    observation, info = environment.reset()
    portfolio_values  = [environment.initial_balance]
    weights_over_time = [environment.current_weights.copy()]
    episode_done      = False

    while not episode_done:
        action, _state   = agent.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = environment.step(action)
        episode_done     = terminated or truncated

        portfolio_values.append(info["portfolio_value"])
        weights_over_time.append(info["weights"].copy())

    daily_returns = []

    for i in range(1, len(portfolio_values)):
        previous_value = portfolio_values[i - 1]
        current_value  = portfolio_values[i]
        daily_return   = (current_value - previous_value) / (previous_value + 1e-10)
        daily_returns.append(daily_return)

    daily_returns = np.array(daily_returns)
    metrics       = compute_metrics(daily_returns)

    print_metrics(agent_name, metrics)

    result = {
        "name"             : agent_name,
        "portfolio_values" : portfolio_values,
        "weights_over_time": weights_over_time,
        "daily_returns"    : daily_returns,
        "metrics"          : metrics,
    }

    return result


def compute_metrics(daily_returns):

    number_of_days = max(len(daily_returns), 1)

    total_return   = float(np.prod(1 + daily_returns) - 1)
    annual_return  = (1 + total_return) ** (252 / number_of_days) - 1
    annual_vol     = float(np.std(daily_returns)) * np.sqrt(252)
    sharpe_ratio   = (annual_return - 0.02) / (annual_vol + 1e-10)

    negative_returns = daily_returns[daily_returns < 0]

    if len(negative_returns) > 0:
        downside_vol  = float(np.std(negative_returns)) * np.sqrt(252)
    else:
        downside_vol  = 1e-10

    sortino_ratio  = (annual_return - 0.02) / downside_vol

    cumulative     = np.cumprod(1 + daily_returns)
    running_peak   = np.maximum.accumulate(cumulative)
    drawdowns      = (running_peak - cumulative) / (running_peak + 1e-10)
    max_drawdown   = float(np.max(drawdowns))

    calmar_ratio   = annual_return / (max_drawdown + 1e-10)
    win_rate       = float(np.mean(daily_returns > 0))

    metrics = {
        "total_return_%"  : round(total_return  * 100, 2),
        "annual_return_%" : round(annual_return * 100, 2),
        "annual_vol_%"    : round(annual_vol    * 100, 2),
        "sharpe_ratio"    : round(sharpe_ratio,        3),
        "sortino_ratio"   : round(sortino_ratio,       3),
        "max_drawdown_%"  : round(max_drawdown  * 100, 2),
        "calmar_ratio"    : round(calmar_ratio,        3),
        "win_rate_%"      : round(win_rate      * 100, 2),
    }

    return metrics


def print_metrics(agent_name, metrics):

    print(f"\n--- {agent_name} ---")

    for metric_name, metric_value in metrics.items():
        print(f"  {metric_name:<20s} : {metric_value}")


def compare_results(list_of_results):

    rows = []

    for result in list_of_results:
        row = {"Agent": result["name"]}
        row.update(result["metrics"])
        rows.append(row)

    comparison_table = pd.DataFrame(rows).set_index("Agent")

    print("\n========== COMPARISON ==========")
    print(comparison_table.to_string())
    print("=================================")

    return comparison_table


def plot_results(list_of_results, assets, initial_balance=100_000.0):

    colors         = ["#1D9E75", "#534AB7", "#BA7517"]
    figure, axes   = plt.subplots(1, 3, figsize=(15, 5))
    figure.suptitle("Portfolio Optimizer - Results", fontsize=14, fontweight="bold")

    for result, color in zip(list_of_results, colors):
        portfolio_values = np.array(result["portfolio_values"])
        indexed_values   = portfolio_values / initial_balance * 100
        axes[0].plot(indexed_values, label=result["name"], color=color, linewidth=2)

    axes[0].axhline(100, color="gray", linestyle="--", linewidth=0.8)
    axes[0].set_title("Portfolio Value (Indexed to 100)")
    axes[0].set_xlabel("Trading Days")
    axes[0].set_ylabel("Value")
    axes[0].legend()

    for result, color in zip(list_of_results, colors):
        portfolio_values = np.array(result["portfolio_values"])
        running_peak     = np.maximum.accumulate(portfolio_values)
        drawdown         = (running_peak - portfolio_values) / (running_peak + 1e-10) * 100

        axes[1].fill_between(range(len(drawdown)), 0, -drawdown, color=color, alpha=0.3, label=result["name"])
        axes[1].plot(-drawdown, color=color, linewidth=0.8)

    axes[1].set_title("Drawdown (%)")
    axes[1].set_xlabel("Trading Days")
    axes[1].set_ylabel("Drawdown (%)")
    axes[1].legend()

    first_result  = list_of_results[0]
    final_weights = np.array(first_result["weights_over_time"][-1])

    axes[2].bar(assets, final_weights, color=colors[0], edgecolor="white")
    axes[2].axhline(1 / len(assets), color="gray", linestyle="--", linewidth=0.8, label="Equal weight")
    axes[2].set_title(f"{first_result['name']} - Final Weights")
    axes[2].set_ylabel("Weight")
    axes[2].set_ylim(0, 0.35)
    axes[2].legend()

    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig("results.png", dpi=150, bbox_inches="tight")
    print("Plot saved to results.png")
    plt.show()