import numpy as np
import pandas as pd
from cvxopt import matrix, solvers


solvers.options["show_progress"] = False


def compute_asset_beta(returns, asset, benchmark):
    """
    Compute beta(asset, benchmark) = Cov(asset, benchmark) / Var(benchmark).
    """
    cov_matrix = np.cov(
        returns[asset].values,
        returns[benchmark].values,
    )

    return cov_matrix[0, 1] / cov_matrix[1, 1]


def compute_beta_vector(returns, tickers_U, benchmark):
    """
    Compute benchmark beta for every asset in the hedge universe.
    """
    beta_dict = {}

    for ticker in tickers_U:
        beta_dict[ticker] = compute_asset_beta(
            returns=returns,
            asset=ticker,
            benchmark=benchmark,
        )

    beta_vec = np.array([beta_dict[ticker] for ticker in tickers_U])

    return beta_dict, beta_vec


def solve_min_variance_target_beta(Sigma, beta_vec, beta_target):
    """
    Solve:

        min_w   w' Sigma w

    subject to:

        beta' w = beta_target
        1' w    = 1
    """
    n = len(beta_vec)

    P = matrix(2 * Sigma)
    q = matrix(np.zeros(n))

    A = matrix(np.vstack([beta_vec, np.ones(n)]))
    b = matrix([beta_target, 1.0])

    sol = solvers.qp(P, q, None, None, A, b)

    return np.array(sol["x"]).flatten()


def solve_min_variance_with_return_tilt(
    Sigma,
    mu,
    beta_vec,
    beta_target,
    lam,
):
    """
    Solve:

        min_w   w' Sigma w - lambda * mu' w

    subject to:

        beta' w = beta_target
        1' w    = 1

    lam = 0 gives the pure minimum-variance problem.
    """
    mu_vec = np.asarray(mu, dtype=float).reshape(-1)
    n = len(mu_vec)

    P = matrix(2 * Sigma)
    q = matrix(-lam * mu_vec)

    A = matrix(np.vstack([beta_vec, np.ones(n)]))
    b = matrix([beta_target, 1.0])

    sol = solvers.qp(P, q, None, None, A, b)

    return np.array(sol["x"]).flatten()


def run_return_tilt_sweep(
    Sigma,
    mu,
    beta_vec,
    beta_target,
    lambdas,
):
    """
    Solve the lambda-tilted Markowitz problem for multiple lambda values.
    """
    mu_vec = np.asarray(mu, dtype=float).reshape(-1)

    weights_by_lambda = {}
    expected_returns = {}

    for lam in lambdas:
        lam_key = f"{lam:.1f}"

        w = solve_min_variance_with_return_tilt(
            Sigma=Sigma,
            mu=mu_vec,
            beta_vec=beta_vec,
            beta_target=beta_target,
            lam=lam,
        )

        weights_by_lambda[lam_key] = w
        expected_returns[lam_key] = float(mu_vec @ w)

    expected_returns = pd.Series(
        expected_returns,
        name="expected_daily_return",
    )

    return weights_by_lambda, expected_returns


def two_fund_combination(w_low, w_high, alpha):
    """
    Combine two minimum-variance portfolios using the two-fund theorem.
    """
    return alpha * w_low + (1 - alpha) * w_high


def solve_target_asset_hedge(
    returns_past_year,
    Sigma,
    tickers_U,
    target,
    benchmark,
):
    """
    Build a hedge portfolio whose beta matches the target asset beta.
    """
    beta_U, beta_vec = compute_beta_vector(
        returns=returns_past_year,
        tickers_U=tickers_U,
        benchmark=benchmark,
    )

    beta_target_asset = compute_asset_beta(
        returns=returns_past_year,
        asset=target,
        benchmark=benchmark,
    )

    w_hedge = solve_min_variance_target_beta(
        Sigma=Sigma,
        beta_vec=beta_vec,
        beta_target=beta_target_asset,
    )

    return {
        "beta_U": beta_U,
        "beta_vec": beta_vec,
        "beta_target_asset": beta_target_asset,
        "weights": w_hedge,
    }


def solve_beta_neutral_expanded_universe(
    returns_past_year,
    tickers_U,
    target,
    benchmark,
):
    """
    Solve beta-neutral minimum-variance portfolio in expanded universe:

        V = U union {target}
    """
    tickers_V = tickers_U + [target]

    returns_V = returns_past_year[tickers_V]
    Sigma_V = returns_V.cov().values.astype(float)

    ridge = 1e-8 * np.trace(Sigma_V) / Sigma_V.shape[0]
    Sigma_V = Sigma_V + ridge * np.eye(Sigma_V.shape[0])

    beta_V_dict, beta_V = compute_beta_vector(
        returns=returns_past_year,
        tickers_U=tickers_V,
        benchmark=benchmark,
    )

    w_beta_neutral = solve_min_variance_target_beta(
        Sigma=Sigma_V,
        beta_vec=beta_V,
        beta_target=0.0,
    )

    return {
        "tickers_V": tickers_V,
        "beta_V_dict": beta_V_dict,
        "beta_V": beta_V,
        "Sigma_V": Sigma_V,
        "weights": w_beta_neutral,
    }


def weights_to_series(weights, tickers, name="weights"):
    """
    Convert weight vector into a readable pandas Series.
    """
    return pd.Series(weights, index=tickers, name=name)