import numpy as np
import pandas as pd


def apply_liquidity_filter(
    raw,
    tickers,
    tickers_U,
    min_obs=200,
    max_missing_ratio=0.05,
    min_avg_price=5,
    min_avg_dollar_volume=20_000_000,
    max_daily_vol=0.10,
    output_path="data/universe/liquidity_filtered_universe.csv",
):
    close = raw["Close"]
    volume = raw["Volume"]

    available_tickers = close.columns[close.notna().any()].tolist()

    print("Downloaded tickers:", len(available_tickers))
    print("Failed or missing tickers:", len([t for t in tickers if t not in available_tickers]))

    close = close[available_tickers]
    volume = volume[available_tickers]

    returns_raw = close.pct_change(fill_method=None)

    quality = pd.DataFrame(index=available_tickers)

    quality["obs_count"] = close.notna().sum()
    quality["missing_ratio"] = close.isna().mean()
    quality["avg_price"] = close.mean()
    quality["avg_volume"] = volume.mean()
    quality["avg_dollar_volume"] = (close * volume).mean()
    quality["daily_vol"] = returns_raw.std()

    candidate_tickers = [t for t in tickers_U if t in quality.index]
    candidate_quality = quality.loc[candidate_tickers].copy()

    candidate_quality["pass_filter"] = (
        (candidate_quality["obs_count"] >= min_obs)
        & (candidate_quality["missing_ratio"] <= max_missing_ratio)
        & (candidate_quality["avg_price"] >= min_avg_price)
        & (candidate_quality["avg_dollar_volume"] >= min_avg_dollar_volume)
        & (candidate_quality["daily_vol"] <= max_daily_vol)
    )

    filtered_quality = candidate_quality[candidate_quality["pass_filter"]].copy()
    filtered_quality = filtered_quality.sort_values("avg_dollar_volume", ascending=False)

    print("Raw hedge universe size:", len(candidate_quality))
    print("Filtered hedge universe size:", len(filtered_quality))
    print("Top 25 filtered candidates:")
    print(filtered_quality.head(25))

    filtered_quality.to_csv(output_path)

    return close, volume, filtered_quality


def rank_hedge_candidates(
    close,
    filtered_quality,
    target,
    benchmark,
    output_path="data/universe/hedge_relevance_ranking.csv",
):
    liquid_tickers = filtered_quality.index.tolist()

    working_tickers = [target, benchmark] + liquid_tickers
    working_tickers = [t for t in working_tickers if t in close.columns]

    returns_filtered = close[working_tickers].pct_change(fill_method=None).dropna()

    R_target = returns_filtered[target]
    R_benchmark = returns_filtered[benchmark]

    ranking = pd.DataFrame(index=liquid_tickers)

    ranking["corr_to_target"] = returns_filtered[liquid_tickers].corrwith(R_target)
    ranking["corr_to_benchmark"] = returns_filtered[liquid_tickers].corrwith(R_benchmark)
    ranking["daily_vol"] = returns_filtered[liquid_tickers].std()

    var_benchmark = R_benchmark.var()
    ranking["beta_to_benchmark"] = [
        returns_filtered[t].cov(R_benchmark) / var_benchmark
        for t in liquid_tickers
    ]

    var_target = R_target.var()
    ranking["beta_to_target"] = [
        returns_filtered[t].cov(R_target) / var_target
        for t in liquid_tickers
    ]

    ranking["avg_dollar_volume"] = filtered_quality["avg_dollar_volume"]

    ranking["score"] = (
        ranking["corr_to_target"].abs() * 0.50
        + ranking["corr_to_benchmark"].abs() * 0.20
        + ranking["avg_dollar_volume"].rank(pct=True) * 0.20
        - ranking["daily_vol"].rank(pct=True) * 0.10
    )

    ranking = ranking.dropna()
    ranking = ranking.sort_values("score", ascending=False)

    print("\nTop 50 hedge relevance candidates:")
    print(ranking.head(50))

    ranking.to_csv(output_path)

    return ranking


def prepare_markowitz_inputs(
    close,
    ranking,
    target,
    benchmark,
    top_n,
    investment_date,
    lookback_start,
):
    tickers_U = ranking.head(top_n).index.tolist()

    print("Final Markowitz hedge universe size:", len(tickers_U))
    print("Final Markowitz hedge universe:")
    print(tickers_U)

    tickers = [target, benchmark] + tickers_U

    data = close[tickers].dropna(how="any")

    returns = data.pct_change(fill_method=None)
    returns = returns.replace([np.inf, -np.inf], np.nan)
    returns = returns.dropna(how="any")

    returns_past_year = returns.loc[lookback_start:investment_date].copy()

    returns_U = returns_past_year[tickers_U].copy()
    returns_U = returns_U.replace([np.inf, -np.inf], np.nan)
    returns_U = returns_U.dropna(axis=1, how="any")

    asset_var = returns_U.var()
    returns_U = returns_U.loc[:, asset_var > 1e-12] 
    #By keeping only assets with variance above the threshold (1e-12), 
    #the code ensures the covariance matrix used in Markowitz optimization 
    #is well-conditioned and invertible.
    #Stocks with zero (or near-zero) variance are problematic for hedging (for example, no risk contribution, undefined correlations, numerical instability)
    

    tickers_U = returns_U.columns.tolist()

    mu_U = returns_U.mean()
    cov_U = returns_U.cov()

    Sigma = cov_U.values.astype(float)

    ridge = 1e-8 * np.trace(Sigma) / Sigma.shape[0]
    Sigma = Sigma + ridge * np.eye(Sigma.shape[0])

    print("\nPre-Markowitz Final Check")
    print("Final universe size:", len(tickers_U))
    print("First 10 hedge candidates:", tickers_U[:10])
    print("Target in universe?", target in tickers_U)
    print("Benchmark in universe?", benchmark in tickers_U)
    print("Target in returns?", target in returns.columns)
    print("Benchmark in returns?", benchmark in returns.columns)
    print("returns shape:", returns.shape)
    print("returns_past_year shape:", returns_past_year.shape)
    print("returns_U shape:", returns_U.shape)
    print("Any missing values in returns_U?", returns_U.isna().sum().sum())
    print("Any NaN in Sigma?", np.isnan(Sigma).any())
    print("Any inf in Sigma?", np.isinf(Sigma).any())
    print("Covariance matrix shape:", Sigma.shape)

    if np.isfinite(Sigma).all():
        print("Covariance matrix rank:", np.linalg.matrix_rank(Sigma))

    return {
        "tickers_U": tickers_U,
        "returns": returns,
        "returns_past_year": returns_past_year,
        "returns_U": returns_U,
        "mu_U": mu_U,
        "cov_U": cov_U,
        "Sigma": Sigma,
    }