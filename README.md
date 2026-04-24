# BasketHedge: Markowitz-Based Equity Hedging

BasketHedge is a portfolio optimization project that uses the Markowitz minimum-variance framework to construct hedging portfolios for a single-stock position. The main example focuses on hedging a long position in AAPL using a basket of ETFs while controlling the portfolio's market beta relative to SPY.

The project is built around the idea that a hedge should not simply reduce exposure in a loose sense. Instead, the hedge is constructed through an explicit optimization problem. I estimate historical returns, covariance matrices, and asset betas, then solve constrained quadratic programs to find portfolios with a target beta.

## Main Idea

The core optimization problem is:

```math
\min_w \quad w^\top \Sigma w
```

subject to the beta constraint:

```math
\beta^\top w = \beta_{\mathrm{target}}
```

and the budget constraint:

```math
\mathbf{1}^\top w = 1
```

where:

- `w` is the vector of portfolio weights
- `Sigma` is the covariance matrix of asset returns
- `beta` is the vector of asset betas relative to SPY
- `beta_target` is the desired portfolio beta

This allows the project to solve for a minimum-variance portfolio under a specific beta constraint.

## What the Project Does

The project is divided into several parts.

## 1. Minimum-Variance Portfolios with Target Beta

I first solve the Markowitz problem for portfolios with different target betas, such as 0.5 and 1.5.

In notation, these portfolios are:

```math
w_{\mathrm{mv}}(0.5)
```

and:

```math
w_{\mathrm{mv}}(1.5)
```

Then I use the two-fund theorem to show that a portfolio with an intermediate beta can be written as a linear combination of two minimum-variance portfolios.

For example:

```math
w_{\mathrm{mv}}(1.0) = 0.5w_{\mathrm{mv}}(0.5) + 0.5w_{\mathrm{mv}}(1.5)
```

This part verifies that the beta constraint is preserved through the linear combination.

## 2. AAPL Hedge Construction

I estimate AAPL's beta relative to SPY using historical daily returns:

```math
\beta_{\mathrm{AAPL}} = \frac{\mathrm{Cov}(R_{\mathrm{AAPL}}, R_{\mathrm{SPY}})}{\mathrm{Var}(R_{\mathrm{SPY}})}
```

Then I construct a hedge portfolio from the ETF universe with the same beta as AAPL:

```math
\beta^\top w_{\mathrm{hedge}} = \beta_{\mathrm{AAPL}}
```

The idea is that shorting this optimized ETF basket against a long AAPL position reduces the market beta of the overall trade.

The hedged return is:

```math
R_{\mathrm{hedged}} = R_{\mathrm{AAPL}} - R_U^\top w_{\mathrm{hedge}}
```

where `R_U` represents the return vector of the ETF hedge universe.

## 3. Beta-Neutral Minimum-Variance Portfolio

I also solve an expanded Markowitz problem where AAPL is included directly in the investment universe.

The enlarged universe is:

```math
V = U \cup \{\mathrm{AAPL}\}
```

In this case, the optimizer finds a beta-neutral portfolio with target beta equal to zero:

```math
\beta_V^\top w = 0
```

The full optimization problem is:

```math
\min_w \quad w^\top \Sigma_V w
```

subject to:

```math
\beta_V^\top w = 0
```

and:

```math
\mathbf{1}^\top w = 1
```

This gives a cleaner comparison between a direct AAPL hedge and a fully optimized beta-neutral portfolio.

## 4. Out-of-Sample Backtest

The hedging strategies are tested out-of-sample using realized returns after the portfolio construction date.

I compare the strategies using:

- cumulative returns
- daily mean return
- volatility
- skewness
- kurtosis
- 95% Value-at-Risk
- realized beta relative to SPY

Cumulative return is computed as:

```math
\prod_{t=1}^{T}(1+r_t)-1
```

Here, `r_1, r_2, ..., r_T` are the strategy returns over each trading interval. In this project, the data is daily, so each `r_t` is one daily return.

The realized beta of each strategy relative to SPY is computed as:

```math
\beta_{\mathrm{strategy}} = \frac{\mathrm{Cov}(R_{\mathrm{strategy}}, R_{\mathrm{SPY}})}{\mathrm{Var}(R_{\mathrm{SPY}})}
```

## 5. Risk-Return Extension

I extend the objective function by adding a linear expected return term:

```math
\min_w \quad w^\top \Sigma w - \lambda \mu^\top w
```

where:

- `mu` is the vector of historical mean returns
- `lambda` controls how much expected return matters in the optimization

When `lambda = 0`, the problem becomes the pure minimum-variance problem.

As `lambda` increases, the optimizer gives more weight to expected return. In the project, this produces more aggressive portfolios with larger swings in realized performance. This shows the tradeoff between minimizing risk and chasing expected return.

## 6. Dynamic Hedging

Finally, I implement a dynamic hedge using a 60-day rolling window and weekly rebalancing.

At each rebalance date, the model re-estimates:

- the covariance matrix
- ETF betas
- AAPL beta
- the optimal hedge weights

Then it solves a new Markowitz problem:

```math
\min_{w_t} \quad w_t^\top \Sigma_t w_t
```

subject to:

```math
\beta_t^\top w_t = \beta_{\mathrm{AAPL},t}
```

and:

```math
\mathbf{1}^\top w_t = 1
```

The dynamic hedge return is:

```math
R_{\mathrm{dynamic},t} = R_{\mathrm{AAPL},t} - R_{U,t}^\top w_t
```

This gives a more realistic version of the hedge because portfolio weights are updated through time instead of remaining fixed.

## ETF Universe

The hedge basket uses the following ETFs:

```python
["FXE", "EWJ", "GLD", "QQQ", "SHV", "DBA", "USO", "XBI", "ILF", "EPP", "FEZ"]
```

These ETFs give exposure to currencies, equities, commodities, bonds, and international markets, which gives the optimizer a broad set of instruments to construct the hedge.

## Tools Used

- Python
- pandas
- NumPy
- yfinance
- cvxopt
- matplotlib

## Key Takeaway

The main takeaway from this project is that hedging a stock position is not just about taking an opposite position in a correlated asset. A better hedge can be built by solving a constrained optimization problem that balances market beta, covariance structure, and realized risk.

The project shows that a simple AAPL hedge can reduce market exposure, but it may still carry meaningful idiosyncratic risk. In contrast, the beta-neutral minimum-variance portfolio is designed to be much more stable and closer to market-neutral behavior. The dynamic hedging extension also shows how the hedge can be updated as market relationships change over time.
