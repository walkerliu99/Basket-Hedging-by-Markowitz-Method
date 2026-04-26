import numpy as np
import pandas as pd
from optimizer import (
    compute_asset_beta,
    compute_beta_vector,
    solve_min_variance_target_beta,
)


def backtest_target_hedge(
    returns_test,
    tickers_U,
    target,
    weights,
    name="Hedged_Target",
):
    """
    Compute realized returns of:

        target return - hedge basket return
    """
    returns_U_test = returns_test[tickers_U]
    target_returns = returns_test[target].values

    hedge_returns = target_returns - returns_U_test.values @ weights

    return pd.Series(
        hedge_returns,
        index=returns_test.index,
        name=name,
    )


def backtest_weighted_portfolio(
    returns_test,
    tickers,
    weights,
    name="Weighted_Portfolio",
):
    """
    Compute realized returns of a weighted portfolio.
    """
    portfolio_returns = returns_test[tickers].values @ weights

    return pd.Series(
        portfolio_returns,
        index=returns_test.index,
        name=name,
    )


def backtest_lambda_sweep(
    returns_test,
    tickers_U,
    target,
    weights_by_lambda,
):
    """
    Backtest hedge returns for a dictionary of lambda-specific weights.
    """
    results = {}

    for lam_key, weights in weights_by_lambda.items():
        series = backtest_target_hedge(
            returns_test=returns_test,
            tickers_U=tickers_U,
            target=target,
            weights=weights,
            name=f"Hedge_lambda_{lam_key}",
        )

        results[lam_key] = series

    df_lam = pd.concat(results.values(), axis=1)
    cum_lam = (1 + df_lam).cumprod() - 1

    return df_lam, cum_lam


def compare_static_hedges(
    returns,
    tickers_U,
    target,
    benchmark,
    w_hedge,
    tickers_V,
    w_beta_neutral,
    test_start,
    test_end,
):
    """
    Compare:
    1. Target asset hedged by a short optimized basket
    2. Beta-neutral minimum-variance portfolio in expanded universe
    """
    returns_test = returns.loc[test_start:test_end].copy()

    hedged_series = backtest_target_hedge(
        returns_test=returns_test,
        tickers_U=tickers_U,
        target=target,
        weights=w_hedge,
        name=f"Hedged_{target}",
    )

    beta_neutral_series = backtest_weighted_portfolio(
        returns_test=returns_test,
        tickers=tickers_V,
        weights=w_beta_neutral,
        name="BetaNeutral",
    )

    results = pd.concat(
        [hedged_series, beta_neutral_series],
        axis=1,
    )

    cumulative_returns = (1 + results).cumprod() - 1
    summary = compute_summary_stats(results)

    benchmark_returns = returns_test[benchmark]

    realized_betas = pd.Series(
        {
            hedged_series.name: compute_realized_beta(
                hedged_series,
                benchmark_returns,
            ),
            beta_neutral_series.name: compute_realized_beta(
                beta_neutral_series,
                benchmark_returns,
            ),
        },
        name=f"Realized beta vs {benchmark}",
    )

    return {
        "returns_test": returns_test,
        "results": results,
        "cumulative_returns": cumulative_returns,
        "summary": summary,
        "realized_betas": realized_betas,
    }


def run_dynamic_target_beta_hedge(
    returns,
    tickers_U,
    target,
    benchmark,
    test_start,
    test_end,
    window=60,
    rebalance_freq=5,
):
    """
    Dynamic rolling target-beta hedge.

    At each rebalance date:
    - use the past rolling window
    - estimate covariance matrix for hedge universe
    - estimate hedge asset betas relative to benchmark
    - estimate target beta relative to benchmark
    - solve min-variance hedge with beta matching target beta
    - apply hedge weights until next rebalance
    """
    returns_test = returns.loc[test_start:test_end].copy()

    if returns_test.empty:
        raise ValueError("returns_test is empty. Check test_start and test_end.")

    test_dates = returns_test.index

    dynamic_returns = []
    weights_history = {}

    current_w = None

    for i, date in enumerate(test_dates):
        if i % rebalance_freq == 0 or current_w is None:
            idx = returns.index.get_loc(date)
            start_idx = max(0, idx - window)

            past = returns.iloc[start_idx:idx].copy()
            needed_cols = [target, benchmark] + tickers_U
            past = past[needed_cols].replace([np.inf, -np.inf], np.nan).dropna(how="any")

            if len(past) < 20:
                raise ValueError(
                    f"Not enough clean historical rows before {date}. "
                    f"Only {len(past)} rows available."
                )

            window_U = past[tickers_U]

            Sigma_temp = window_U.cov().values.astype(float)

            ridge = 1e-8 * np.trace(Sigma_temp) / Sigma_temp.shape[0]
            Sigma_temp = Sigma_temp + ridge * np.eye(Sigma_temp.shape[0])

            _, beta_vec_temp = compute_beta_vector(
                returns=past,
                tickers_U=tickers_U,
                benchmark=benchmark,
            )

            beta_target_temp = compute_asset_beta(
                returns=past,
                asset=target,
                benchmark=benchmark,
            )

            current_w = solve_min_variance_target_beta(
                Sigma=Sigma_temp,
                beta_vec=beta_vec_temp,
                beta_target=beta_target_temp,
            )

            weights_history[date] = pd.Series(
                current_w,
                index=tickers_U,
                name=date,
            )

        target_return_t = returns.loc[date, target]
        hedge_return_t = returns.loc[date, tickers_U].values @ current_w

        dynamic_return_t = target_return_t - hedge_return_t
        dynamic_returns.append(dynamic_return_t)

    dynamic_hedge_series = pd.Series(
        dynamic_returns,
        index=test_dates,
        name="DynamicHedge",
    )

    weights_history = pd.DataFrame(weights_history).T

    return dynamic_hedge_series, weights_history


def compute_summary_stats(results):
    """
    Compute basic performance statistics for strategy returns.
    """
    return pd.DataFrame(
        {
            "Mean": results.mean(),
            "Volatility": results.std(),
            "Skewness": results.skew(),
            "Kurtosis": results.kurtosis(),
            "VaR_95": results.quantile(0.05),
        }
    )


def compute_realized_beta(strategy_returns, benchmark_returns):
    """
    Compute realized beta of strategy returns relative to benchmark returns.
    """
    aligned = pd.concat([strategy_returns, benchmark_returns], axis=1).dropna()

    if aligned.shape[0] < 2:
        raise ValueError("Not enough observations to compute realized beta.")

    c = np.cov(aligned.iloc[:, 0].values, aligned.iloc[:, 1].values)

    return c[0, 1] / c[1, 1]