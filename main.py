# %% Imports

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from universe import load_csv_symbol_universe, get_ticker_list
from data_loader import load_or_download_prices
from preprocessing import (
    apply_liquidity_filter,
    rank_hedge_candidates,
    prepare_markowitz_inputs,
)
from optimizer import (
    compute_beta_vector,
    solve_min_variance_target_beta,
    two_fund_combination,
    solve_target_asset_hedge,
    solve_beta_neutral_expanded_universe,
    run_return_tilt_sweep,
    weights_to_series,
)
from backtest import (
    compare_static_hedges,
    backtest_lambda_sweep,
    run_dynamic_target_beta_hedge,
    compute_summary_stats,
    compute_realized_beta,
)
from visualization import (
    render_table,
    plot_lambda_sweep,
    plot_strategy_comparison,
)

plt.style.use("dark_background")


# %% Step 0: Config

target = "AAPL"
benchmark = "SPY"

start_date = "2024-01-01"
end_date = "2026-02-28"

investment_date = "2025-01-02"
lookback_start = "2024-01-01"

test_start = "2025-01-02"
test_end = "2025-04-02"

top_n = 30

batch_size = 100
pause_seconds = 10

lam_list = [i * 0.1 for i in range(0, 11)]


# %% Step 1: Load CSV Symbol Universe

universe_df = load_csv_symbol_universe(
    etf_file="core_etfs.csv",
    stock_file="custom_universe.csv",
    target=target,
    benchmark=benchmark,
    include_target=False,
    include_benchmark=False,
    save_files=True,
)

tickers_U = get_ticker_list(universe_df)
tickers = [target, benchmark] + tickers_U

print("CSV candidate universe size:", len(tickers_U))
print("First 25 hedge candidates:", tickers_U[:25])
print("Target in universe?", target in tickers_U)
print("Benchmark in universe?", benchmark in tickers_U)
print("\nAsset type counts:")
print(universe_df["asset_type"].value_counts())


# %% Step 2: Load or Download Price / Volume Data

raw, successful_tickers, failed_tickers = load_or_download_prices(
    tickers=tickers,
    target=target,
    benchmark=benchmark,
    start_date=start_date,
    end_date=end_date,
    batch_size=batch_size,
    pause_seconds=pause_seconds,
    auto_adjust=False,
)

print("Successful tickers:", len(successful_tickers))
print("Failed tickers:", len(failed_tickers))
print("First 25 failed:", failed_tickers[:25])


# %% Step 3: Data Quality, Liquidity Filtering, and Candidate Ranking

close, volume, filtered_quality = apply_liquidity_filter(
    raw=raw,
    tickers=tickers,
    tickers_U=tickers_U,
)

ranking = rank_hedge_candidates(
    close=close,
    filtered_quality=filtered_quality,
    target=target,
    benchmark=benchmark,
)

prepared = prepare_markowitz_inputs(
    close=close,
    ranking=ranking,
    target=target,
    benchmark=benchmark,
    top_n=top_n,
    investment_date=investment_date,
    lookback_start=lookback_start,
)

tickers_U = prepared["tickers_U"]
returns = prepared["returns"]
returns_past_year = prepared["returns_past_year"]
returns_U = prepared["returns_U"]
mu_U = prepared["mu_U"]
cov_U = prepared["cov_U"]
Sigma = prepared["Sigma"]

print("\nReady for Markowitz.")
print("Final hedge universe size:", len(tickers_U))
print("returns_past_year shape:", returns_past_year.shape)
print("Sigma shape:", Sigma.shape)


# %% Step 4: Markowitz Two-Fund Theorem Demo

beta_U, beta_vec = compute_beta_vector(
    returns=returns_past_year,
    tickers_U=tickers_U,
    benchmark=benchmark,
)

beta_a = 0.5
beta_b = 1.5
alpha = 0.5

w_a = solve_min_variance_target_beta(
    Sigma=Sigma,
    beta_vec=beta_vec,
    beta_target=beta_a,
)

w_b = solve_min_variance_target_beta(
    Sigma=Sigma,
    beta_vec=beta_vec,
    beta_target=beta_b,
)

w_c = two_fund_combination(
    w_low=w_a,
    w_high=w_b,
    alpha=alpha,
)

print("\nw_mv(0.5):")
print(weights_to_series(w_a, tickers_U, name="w_mv_0.5"))

print("\nw_mv(1.5):")
print(weights_to_series(w_b, tickers_U, name="w_mv_1.5"))

print("\nw_mv(1.0) computed via two-fund theorem:")
print(weights_to_series(w_c, tickers_U, name="w_mv_1.0"))

print("\nBeta checks:")
print("beta(0.5) portfolio beta =", np.dot(beta_vec, w_a))
print("beta(1.5) portfolio beta =", np.dot(beta_vec, w_b))
print("beta(1.0) portfolio beta =", np.dot(beta_vec, w_c))


# %% Step 5: Target Asset Hedge

hedge_result = solve_target_asset_hedge(
    returns_past_year=returns_past_year,
    Sigma=Sigma,
    tickers_U=tickers_U,
    target=target,
    benchmark=benchmark,
)

beta_U = hedge_result["beta_U"]
beta_vec = hedge_result["beta_vec"]
beta_target_asset = hedge_result["beta_target_asset"]
w_hedge = hedge_result["weights"]

print(f"\nBeta of {target} relative to {benchmark}:")
print(beta_target_asset)

print(f"\nHedging portfolio w_mv(beta_{target}):")
print(weights_to_series(w_hedge, tickers_U, name="w_hedge"))

print("\nBeta of hedging portfolio:")
print(np.dot(beta_vec, w_hedge))


# %% Step 6: Beta-Neutral Portfolio in Expanded Universe

beta_neutral_result = solve_beta_neutral_expanded_universe(
    returns_past_year=returns_past_year,
    tickers_U=tickers_U,
    target=target,
    benchmark=benchmark,
)

tickers_V = beta_neutral_result["tickers_V"]
beta_V = beta_neutral_result["beta_V"]
w_beta_neutral = beta_neutral_result["weights"]

print("\nBeta-neutral MVP in expanded universe V:")
print(weights_to_series(w_beta_neutral, tickers_V, name="w_beta_neutral"))

print("\nBeta of beta-neutral portfolio:")
print(np.dot(beta_V, w_beta_neutral))


# %% Step 7: Static Hedge Backtest

static_backtest = compare_static_hedges(
    returns=returns,
    tickers_U=tickers_U,
    target=target,
    benchmark=benchmark,
    w_hedge=w_hedge,
    tickers_V=tickers_V,
    w_beta_neutral=w_beta_neutral,
    test_start=test_start,
    test_end=test_end,
)

returns_test = static_backtest["returns_test"]
results = static_backtest["results"]
cum_returns = static_backtest["cumulative_returns"]
summary = static_backtest["summary"]
realized_betas = static_backtest["realized_betas"]

print("\nStatic hedge return sample:")
print(results.head())

render_table(
    summary,
    f"Performance Summary: Hedged {target} vs Beta-Neutral",
)

print("\nRealized betas:")
print(realized_betas)

plot_strategy_comparison(
    cum_returns=cum_returns,
    title=f"Cumulative Returns: Hedged {target} vs Beta-Neutral",
)


# %% Step 8: Lambda Risk-Return Extension

weights_by_lambda, expected_returns_lambda = run_return_tilt_sweep(
    Sigma=Sigma,
    mu=mu_U,
    beta_vec=beta_vec,
    beta_target=beta_target_asset,
    lambdas=lam_list,
)

print("\nExpected daily return by lambda:")
print(expected_returns_lambda)

df_lam, cum_lam = backtest_lambda_sweep(
    returns_test=returns_test,
    tickers_U=tickers_U,
    target=target,
    weights_by_lambda=weights_by_lambda,
)

print("\nLambda hedge return sample:")
print(df_lam.head())

plot_lambda_sweep(
    cum_lam=cum_lam,
    title="Cumulative Returns for Different Lambda Values",
    ylim=(-0.4, 0.4),
)

col0 = "Hedge_lambda_0.0"
final_0 = cum_lam[col0].iloc[-1]

print("Final return for lambda = 0.0:", final_0)
print("\nLambdas outperforming lambda = 0.0:")

for lam in lam_list:
    lam_key = f"{lam:.1f}"
    col = f"Hedge_lambda_{lam_key}"

    if lam == 0:
        continue

    final_lam = cum_lam[col].iloc[-1]

    if final_lam > final_0:
        print(f"lambda = {lam_key} --> outperforms, final return = {final_lam:.4f}")


# %% Step 9: Dynamic Rolling Hedge

dynamic_hedge_series, dynamic_weights_history = run_dynamic_target_beta_hedge(
    returns=returns,
    tickers_U=tickers_U,
    target=target,
    benchmark=benchmark,
    test_start=test_start,
    test_end=test_end,
    window=60,
    rebalance_freq=5,
)

results_all = pd.concat([results, dynamic_hedge_series], axis=1)

cum_all = (1 + results_all).cumprod() - 1

plot_strategy_comparison(
    cum_returns=cum_all,
    title="Cumulative Returns: Static Hedge vs Beta-Neutral vs Dynamic Hedge",
)

summary_all = compute_summary_stats(results_all)

render_table(
    summary_all,
    "Performance Summary: Static Hedge vs Dynamic Hedge vs Beta-Neutral",
)

beta_dynamic = compute_realized_beta(
    strategy_returns=dynamic_hedge_series,
    benchmark_returns=returns.loc[dynamic_hedge_series.index, benchmark],
)

print("\nDynamic hedge beta vs benchmark:", beta_dynamic)