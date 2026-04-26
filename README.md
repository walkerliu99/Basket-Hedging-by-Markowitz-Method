# BasketHedge: Markowitz-Based Equity Basket Hedging

BasketHedge is a Python-based portfolio optimization project for constructing hedge baskets against a large equity or ETF position. The project started as a portfolio theory class project, but I redesigned it into a reusable research framework with separate modules for universe construction, data loading, preprocessing, optimization, backtesting, and visualization.

The core idea is simple: if an investor needs to hedge or gradually off-load a large position in a stock or ETF, the model searches through a candidate universe of stocks and ETFs, filters and ranks hedge candidates, and then solves constrained Markowitz optimization problems to construct hedge portfolios.

This project is especially relevant for hedging large positions in the equity and ETF market. A basket hedge can reduce dependence on a single hedge instrument, distribute exposure across multiple liquid names, and potentially reduce execution cost and market impact compared with putting the entire hedge into one instrument. It also gives the optimizer more instruments to match the target asset's beta and covariance structure.

---

## Project Motivation

Suppose an investor holds a large long position in a single stock, such as AAPL. A simple hedge would be to short SPY or QQQ. That may reduce broad market exposure, but it does not necessarily hedge the stock's more detailed risk structure.

For example, AAPL may have exposure to technology, growth, semiconductors, rates, broad-market beta, and idiosyncratic stock-specific risk. A single ETF hedge may be too coarse.

BasketHedge tries to build a more systematic hedge by using a candidate investment universe, estimating historical returns and covariances, computing benchmark betas, and solving constrained quadratic programs.

The project focuses on:

- constructing a tradable hedge universe
- filtering candidates by liquidity and data quality
- ranking candidates by hedge relevance
- solving Markowitz-style constrained optimization problems
- comparing static hedge, beta-neutral portfolio, lambda-tilted hedge, and dynamic rolling hedge strategies
- visualizing realized out-of-sample performance

---

## Why Use a Basket Hedge?

A basket hedge can be useful when hedging a large equity position because it may:

- reduce execution cost by spreading trades across multiple liquid names
- reduce market impact compared with trading a very large amount in one hedge instrument
- avoid relying entirely on one ETF or index future
- better match the target asset's covariance and beta structure
- give exposure to sector, factor, rates, commodity, international, and credit instruments
- create a more flexible hedge than a simple single-name or single-ETF hedge

However, basket hedging also introduces challenges:

- more instruments increase transaction costs and operational complexity
- short positions may involve borrow constraints
- Markowitz optimization is sensitive to covariance estimation error
- too many assets relative to observations can produce unstable weights
- highly correlated assets may lead to near-singular covariance matrices
- unconstrained optimization can generate unrealistic long-short weights

Because of these issues, this project does not pass the full raw universe directly into the optimizer. Instead, it first filters and ranks the universe, then sends a smaller final candidate set into the Markowitz optimizer.

---

## Repository Structure

The repo is organized as a research pipeline:

```text
basket-beta-hedging/
│
├── main.py
├── universe.py
├── data_loader.py
├── preprocessing.py
├── optimizer.py
├── backtest.py
├── visualization.py
│
├── data/
│   ├── universe/
│   └── prices/
│
└── README.md
```

---

## Workflow Overview

The full pipeline is:

```text
1. Set project configuration
2. Load the investment universe
3. Download or load cached market data
4. Apply data-quality and liquidity filters
5. Rank hedge candidates
6. Prepare Markowitz inputs
7. Solve optimization problems
8. Backtest hedging strategies
9. Visualize results
```

The user can configure:

- target stock or ETF
- benchmark
- data start date
- data end date
- investment decision date
- lookback window
- hedge backtest date range
- number of final hedge candidates
- batch download size
- yfinance pause time
- lambda values for the risk-return extension

Example configuration:

```python
target = "AAPL"
benchmark = "SPY"

start_date = "2025-01-01"
end_date = "2026-02-28"

investment_date = "2026-01-02"
lookback_start = "2025-01-02"

test_start = "2026-01-05"
test_end = "2026-02-27"

top_n = 30
```

---

## Module Descriptions

### `main.py`

`main.py` is the main driver script. It controls the full workflow, from loading the universe to running the optimization and backtest.

It is intentionally kept as a high-level script. Most of the actual logic lives in the supporting modules.

---

### `universe.py`

`universe.py` handles ticker universe construction.

In the current version, the universe is built from:

```text
S&P 500 stocks + selected liquid ETFs
```

The ETF list includes broad-market ETFs, sector ETFs, rates ETFs, credit ETFs, commodity ETFs, international ETFs, and selected high-volume ETFs.

This module:

- loads stock and ETF symbols
- cleans ticker formatting
- removes duplicate symbols
- excludes the target asset and benchmark from the hedge universe
- returns a clean candidate universe

---

### `data_loader.py`

`data_loader.py` handles price and volume data loading.

Because `yfinance` can rate-limit large downloads, the data loader downloads tickers in batches and caches the raw output locally. This avoids repeatedly hitting the Yahoo Finance API during development.

The output is a raw market data object containing price and volume data.

---

### `preprocessing.py`

`preprocessing.py` is one of the most important modules in the project.

It performs:

```text
Raw universe
→ data-quality filter
→ liquidity filter
→ hedge relevance ranking
→ final hedge universe
→ covariance matrix preparation
```

The filtering step checks:

- number of valid observations
- missing data ratio
- average price
- average volume
- average dollar volume
- daily volatility

The ranking step scores candidates based on:

- correlation with the target asset
- correlation with the benchmark
- average dollar volume
- daily volatility

This matters because the project is not simply throwing hundreds of assets into a Markowitz optimizer. It first constructs a cleaner and more defensible hedge universe.

---

### `optimizer.py`

`optimizer.py` contains the Markowitz optimization logic.

It computes:

- benchmark betas
- target asset beta
- minimum-variance portfolios with target beta constraints
- target-asset hedge basket
- beta-neutral expanded-universe portfolio
- lambda-tilted risk-return portfolios

---

### `backtest.py`

`backtest.py` converts optimized weights into realized strategy returns.

It contains:

- static target hedge backtest
- beta-neutral portfolio backtest
- lambda sweep backtest
- dynamic rolling hedge backtest
- summary statistics
- realized beta calculation

---

### `visualization.py`

`visualization.py` contains plotting and table-rendering functions.

It creates:

- performance summary tables
- cumulative return plots
- lambda sweep plots
- static vs dynamic hedge comparison plots

---

## Data Universe and Current Limitation

The current implementation uses S&P 500 companies and a selected set of liquid ETFs.

This was a practical design choice because `yfinance` limits large data pull requests, and many full-market ticker files contain stale symbols, delisted names, preferred shares, warrants, rights, or symbols that do not map cleanly to Yahoo Finance.

A future version could use the entire U.S. equity market and ETF market if better market data is available. For example, with a professional data source such as Bloomberg, Refinitiv, CRSP, Polygon, WRDS, IEX, Tiingo, or another institutional data vendor, the universe could be expanded to:

```text
Entire U.S. common stock universe + full ETF universe
```

The current framework is designed so that the universe module can be replaced without rewriting the optimizer or backtest logic.

---

## Mathematical Framework

Let:

- $U$ be the final hedge universe.
- $w$ be the vector of hedge portfolio weights.
- $\Sigma$ be the covariance matrix of asset returns in $U$.
- $\beta$ be the vector of betas of assets in $U$ relative to the benchmark.
- $\mu$ be the vector of expected returns.
- $R_{\mathrm{target}}$ be the target asset return.
- $R_U$ be the vector of hedge universe returns.
- $R_{\mathrm{benchmark}}$ be the benchmark return.

The benchmark beta of asset $i$ is estimated as:

$$
\beta_i = \frac{\mathrm{Cov}(R_i, R_{\mathrm{benchmark}})}{\mathrm{Var}(R_{\mathrm{benchmark}})}
$$

The target asset beta is estimated as:

$$
\beta_{\mathrm{target}} = \frac{\mathrm{Cov}(R_{\mathrm{target}}, R_{\mathrm{benchmark}})}{\mathrm{Var}(R_{\mathrm{benchmark}})}
$$

---

## Optimization Problem 1: Target-Beta Minimum-Variance Portfolio

The first optimization problem solves for the minimum-variance portfolio in the hedge universe subject to a target beta constraint and a full-investment constraint.

The objective function is:

$$
\min_w \quad w^\top \Sigma w
$$

subject to:

$$
\beta^\top w = \beta_{\mathrm{target}}
$$

and:

$$
\mathbf{1}^\top w = 1
$$

This gives the minimum-variance portfolio among all portfolios whose benchmark beta equals the chosen target beta.

In the project, I first solve this problem for example beta targets such as $0.5$ and $1.5$. Then I use the two-fund theorem to show that an intermediate target-beta portfolio can be represented as a linear combination of two minimum-variance portfolios.

Let:

$$
w_a = w_{\mathrm{mv}}(\beta_a)
$$

and:

$$
w_b = w_{\mathrm{mv}}(\beta_b)
$$

Then a portfolio with an intermediate beta can be written as:

$$
w_c = \alpha w_a + (1-\alpha)w_b
$$

Its beta is:

$$
\beta^\top w_c = \alpha\beta_a + (1-\alpha)\beta_b
$$

This section verifies that the constrained Markowitz setup behaves consistently with the two-fund theorem.

---

## Optimization Problem 2: Target Asset Hedge Basket

The second optimization problem uses the same target-beta minimum-variance framework, but the target beta is chosen to match the beta of the asset being hedged.

For example, if the target asset is AAPL and the benchmark is SPY, I estimate:

$$
\beta_{\mathrm{AAPL}} = \frac{\mathrm{Cov}(R_{\mathrm{AAPL}}, R_{\mathrm{SPY}})}{\mathrm{Var}(R_{\mathrm{SPY}})}
$$

Then the hedge basket is obtained by solving:

$$
\min_w \quad w^\top \Sigma_U w
$$

subject to:

$$
\beta_U^\top w = \beta_{\mathrm{AAPL}}
$$

and:

$$
\mathbf{1}^\top w = 1
$$

The realized hedged return is:

$$
R_{\mathrm{hedged}} = R_{\mathrm{AAPL}} - R_U^\top w_{\mathrm{hedge}}
$$

The idea is that the hedge basket is constructed to have approximately the same benchmark beta as the target asset. Shorting the optimized basket against the long target position should reduce the systematic beta exposure of the overall trade.

This approach is more flexible than simply shorting SPY or QQQ. It allows the optimizer to choose a diversified basket of instruments that matches the target asset's beta while minimizing the variance of the hedge basket.

---

## Optimization Problem 3: Beta-Neutral Portfolio in Expanded Universe

The third optimization problem expands the investment universe to include the target asset itself.

Let:

$$
V = U \cup \{\mathrm{target}\}
$$

The optimizer solves:

$$
\min_w \quad w^\top \Sigma_V w
$$

subject to:

$$
\beta_V^\top w = 0
$$

and:

$$
\mathbf{1}^\top w = 1
$$

This produces a beta-neutral minimum-variance portfolio in the expanded universe.

Unlike the direct hedge in Problem 2, this optimizer is allowed to choose both the target asset and all hedge instruments simultaneously. The goal is not simply to hedge a pre-existing long position. Instead, the goal is to construct a fully optimized beta-neutral portfolio.

The realized return is:

$$
R_{\mathrm{beta\ neutral}} = R_V^\top w_V
$$

This portfolio is designed to have near-zero benchmark beta while minimizing total variance.

---

## Risk-Return Extension: Lambda-Tilted Markowitz Portfolio

I also include a risk-return extension by adding a linear expected return term to the objective.

The optimization problem becomes:

$$
\min_w \quad w^\top \Sigma w - \lambda \mu^\top w
$$

subject to:

$$
\beta^\top w = \beta_{\mathrm{target}}
$$

and:

$$
\mathbf{1}^\top w = 1
$$

Here, $\lambda$ controls the tradeoff between variance minimization and expected return.

When:

$$
\lambda = 0
$$

the problem reduces to the pure minimum-variance target-beta portfolio.

As $\lambda$ increases, the optimizer gives more weight to expected return. In practice, this can produce more aggressive portfolios with larger long-short exposures and larger realized swings. This part of the project helps illustrate how fragile expected-return inputs can be in portfolio optimization.

---

## Dynamic Rolling Hedge

The dynamic hedge repeats the target-beta hedge construction over time.

At each rebalance date, the model:

1. Looks back over a rolling estimation window.
2. Re-estimates the covariance matrix.
3. Re-estimates hedge candidate betas.
4. Re-estimates target asset beta.
5. Solves the target-beta minimum-variance hedge problem.
6. Applies the hedge weights until the next rebalance.

The dynamic hedge return at time $t$ is:

$$
R_{\mathrm{dynamic},t} = R_{\mathrm{target},t} - R_{U,t}^\top w_t
$$

where $w_t$ is the hedge basket solved using only information available before time $t$.

This is more realistic than a static hedge because market relationships change over time. However, it also creates turnover, transaction costs, and additional estimation error.

---

## Covariance Matrix Sensitivity

A major limitation of Markowitz optimization is its dependence on the covariance matrix.

The optimizer relies heavily on:

$$
\Sigma = \mathrm{Cov}(R_U)
$$

If $\Sigma$ is poorly estimated, the optimizer can produce unstable or unrealistic weights.

This matters because covariance estimation becomes difficult when the number of assets is large relative to the number of observations. For example, if the model uses 50 hedge candidates but only around 250 daily observations, the covariance matrix is mathematically estimable, but it may still be noisy. If the model tried to optimize over hundreds or thousands of assets with the same lookback window, the covariance matrix could become singular, unstable, or dominated by estimation noise.

This is one reason the project filters and ranks candidates before optimization. The model uses a broad search universe, but only sends a smaller final universe into the Markowitz optimizer.

Potential improvements include:

- shrinkage covariance estimation
- factor-model covariance estimation
- robust covariance estimators
- principal component reduction
- maximum weight constraints
- gross exposure constraints
- turnover penalties
- long-only or limited-short constraints
- transaction-cost-aware optimization

---

## Hedging Methodology and Limitations

The main hedging method used in this project is a beta-matching basket hedge.

For a long target asset position, the hedge return is:

$$
R_{\mathrm{hedged}} = R_{\mathrm{target}} - R_U^\top w_{\mathrm{hedge}}
$$

This can reduce benchmark beta exposure, but it does not eliminate all risk. The target asset may still experience idiosyncratic shocks that are not captured by the hedge basket. For example, an earnings surprise, regulatory news, product announcement, or company-specific event can move the target stock even if the hedge basket is well constructed.

Important limitations:

- beta is estimated from historical data and may change out-of-sample
- covariance relationships can break during stress periods
- the hedge may reduce systematic risk but not idiosyncratic risk
- unconstrained Markowitz weights may be hard to implement in real trading
- transaction costs and borrow costs are not fully modeled
- leveraged and inverse ETFs may introduce path dependency and decay
- expected returns are noisy and can make lambda-tilted portfolios unstable
- using current S&P 500 constituents may introduce survivorship bias
- yfinance data is suitable for research but not production trading

---

## Potential Improvements

Future versions of this project could improve the hedge construction by adding the following extensions.

### Transaction-Cost-Aware Optimization

The objective could penalize turnover:

$$
\min_w \quad w^\top \Sigma w + c^\top |w - w_{\mathrm{prev}}|
$$

This would make the dynamic hedge more realistic because frequent rebalancing can be expensive.

### Gross Exposure Constraint

To prevent unrealistic leverage:

$$
\sum_i |w_i| \leq L
$$

This helps control the size of long-short positions.

### Maximum Weight Constraint

To avoid concentration in one hedge instrument:

$$
|w_i| \leq w_{\max}
$$

This prevents the optimizer from placing too much weight on one asset.

### Shorting Constraints

If shorting is limited, one can impose:

$$
w_i \geq 0
$$

or allow only limited short exposure:

$$
w_i \geq -s_{\max}
$$

### Factor-Model Hedge

Instead of matching only benchmark beta, the hedge could match multiple factor exposures:

$$
B^\top w = b_{\mathrm{target}}
$$

where $B$ contains exposures to market, size, value, momentum, quality, rates, or sector factors.

### Shrinkage Covariance

A shrinkage covariance matrix could be used:

$$
\Sigma_{\mathrm{shrunk}} = \delta F + (1-\delta)\Sigma_{\mathrm{sample}}
$$

where $F$ is a structured covariance target and $\delta$ is the shrinkage intensity.

This can make the optimizer less sensitive to noisy sample covariance estimates.

---

## Current Status

The current version supports:

- configurable target stock and benchmark
- configurable data range
- configurable investment decision date
- configurable lookback period
- configurable hedge backtest period
- cached yfinance data loading
- liquidity filtering
- hedge relevance ranking
- target-beta Markowitz optimization
- beta-neutral expanded-universe optimization
- lambda risk-return extension
- dynamic rolling hedge
- performance summaries
- cumulative return plots

---

## Key Takeaway

This project shows that hedging a large equity position is not simply about shorting a broad index ETF. A more systematic hedge can be constructed by filtering a tradable universe, ranking hedge candidates, estimating covariance and beta relationships, and solving constrained Markowitz optimization problems.

The main strength of the project is the full pipeline: universe construction, liquidity filtering, hedge relevance ranking, optimization, and out-of-sample backtesting.

The main limitation is that Markowitz optimization is highly sensitive to covariance estimation and implementation assumptions. Therefore, the hedge should be interpreted as a research framework rather than a production trading system.
