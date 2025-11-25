import yfinance as yf
import pandas as pd
import numpy as np
from cvxopt import matrix, solvers
import matplotlib.pyplot as plt

#Preliminary Step
tickers_U = ["FXE","EWJ","GLD","QQQ","SHV","DBA","USO","XBI","ILF","EPP","FEZ"]
tickers = ["AAPL","SPY"] + tickers_U
start_date = "2024-01-01"
end_date = "2025-02-28"

raw = yf.download(tickers, start=start_date, end=end_date)

data = raw["Close"]
data = data.dropna()

returns = data.pct_change().dropna()

#assume that the investment decision is taken at the end of trading on January 2nd, 2025 and that the set U is considered as your investment.
#Use 1 year of historical data to compute the Expected Returns and Covariance needed for the optimization.
# We must use only data up to the investment date: January 2, 2025
cutoff = "2025-01-02"

# Use only the past 1 year of data (automatically cuts from 2024-01-02 to 2025-01-02)
returns_past_year = returns.loc["2024-01-02":cutoff]

# Extract U universe
returns_U = returns_past_year[tickers_U]

# Expected returns and covariance from THIS window ONLY
mu_U = returns_U.mean()
cov_U = returns_U.cov()


#Motivation:
#We want to solve the Markowitz problem with constraint Aw = b, where w is the min-variance portfolio with weights
#Then apply the two-fund theorem

#------------------------------------------------------------Part 1--------------------------------------------------------------------------------
#Toy scenario:
#given beta_a = 0.5, beta_b = 1.5, beta_c = 1, and beta_c = a*(beta_a) + (1-a)*beta_b. Then a = 0.5
#Let w_mv(beta_a) be the min-variance portfolio with target beta 0.5, and let w_mv(beta_b) be the min-variance portfolio with target beta 1.5
#By two-fund theorem, w_mv(beta_c) is a linear combination of w_mv(beta_a) and w_mv(beta_b) with corresponding sclars a and (1-a)
#Here we have w_mv(1) = 0.5*w_mv(0.5) + 0.5*w_mv(1.5)

#Markowitz Problem
#parameteres
returns_SPY = returns_past_year["SPY"]  

#Compute betas for the ETFs in U relative to SPY
beta_U = {}
for t in tickers_U:
    #for each iteration of the calculation, we compute the 2-by-2 covariance matrix with daily returns of each ticker in U w.r.t SPY daily returns
    cov_ts = np.cov(returns_past_year[t].values, returns_SPY.values) 
    #recall that beta_U[t] is just "Cov(t, SPY)/Var(SPY)", so we pull the data from our 2-by-2 covariance matrix and do the computaion
    beta_U[t] = cov_ts[0,1] / cov_ts[1,1] 

beta_vec = np.array([beta_U[t] for t in tickers_U]) #after each iteration, we store our beta_U in vector form

#Setup the Markowitz QP using CVXOPT
#Covariance matrix and dimensionality
Sigma = cov_U.values
n = len(tickers_U)

# QP matrices
P = matrix(2 * Sigma)       # this cvxopt solves problems in the form min_w(1/2 * w^T P w + q^T w), we set P = 2*Sigma
q = matrix(np.zeros(n))     # for now let's set q^T = zero matrix

def solve_mv(beta_target):
    #Solves: min w^T Sigma w  s.t.  beta_w = beta_target AND weights of the portfolio sum to 1
    # Equality constraints: A w = b
    A = matrix(np.vstack([beta_vec, np.ones(n)]))
    b = matrix([beta_target, 1.0])

    # No inequality constraints, so G, h = None
    sol = solvers.qp(P, q, None, None, A, b)
    w = np.array(sol['x']).flatten()
    return w

beta_a = 0.5
beta_b = 1.5

w_a = solve_mv(beta_a)    # ω_mv(0.5)
w_b = solve_mv(beta_b)    # ω_mv(1.5)

print("\nw_mv(0.5):")
print(pd.Series(w_a, index=tickers_U))

print("\nw_mv(1.5):")
print(pd.Series(w_b, index=tickers_U))

#Apply Two-fund theorem where α = 0.5
alpha = 0.5
w_c = alpha * w_a + (1 - alpha) * w_b

print("\nw_mv(1.0) computed via two-fund theorem:")
print(pd.Series(w_c, index=tickers_U))

#verify the beta constraint
print("beta(0.5) portfolio beta =", np.dot(beta_vec, w_a))
print("beta(1.5) portfolio beta =", np.dot(beta_vec, w_b))
print("beta(1.0) portfolio beta =", np.dot(beta_vec, w_c))


#----------------------------------------------------------------Part 2----------------------------------------------------------------------------
#toy scenario: assume that the investment decision is taken at the end of trading on January 2nd, 2025, say we want to hedge a long position in Apple only using products in U
#Considering the enlarged investment universe V = U \union {AAPL},find in that universe the Minimum Variance Portfolio with Target Beta equal to 0. (Beta neutral Portfolio)

#(a)First compute beta_AAPL
cov_aapl_spy = np.cov(returns_past_year["AAPL"].values, returns_SPY.values)
beta_AAPL = cov_aapl_spy[0,1] / cov_aapl_spy[1,1]

print("\nBeta of AAPL relative to SPY:")
print(beta_AAPL)

# Combine into a dictionary
beta_all = {"AAPL": beta_AAPL, "SPY": 1.0}
beta_all.update(beta_U)

print("\nAll betas (AAPL, SPY, and U):")
print(beta_all)

#(b)constructing the hedging portfolio
w_hedge = solve_mv(beta_AAPL)

print("\nHedging portfolio ω_mv(beta_AAPL):")
print(pd.Series(w_hedge, index=tickers_U))

# Verify the beta constraint holds
print("\nBeta of hedging portfolio (should equal beta_AAPL):")
print(np.dot(beta_vec, w_hedge))

#Interpretation (Part2(b)): we set the target beta for the Markowitz problem the same as the beta of Apple. Then, to hedge our long position in Apple, we just need to -w_hedge units of the tickers in U

#(c)Beta-neutral min-variance portfolio in enlarged universe V = U \union {AAPL}
tickers_V = tickers_U + ["AAPL"]

returns_V = returns_past_year[tickers_V]
mu_V = returns_V.mean()
cov_V = returns_V.cov()

beta_V = np.array([beta_U[t] for t in tickers_U] + [beta_AAPL]) #build the expanded beta vector

#Markowitz problem setup for the enlarged universe
Sigma_V = cov_V.values
n_V = len(tickers_V)

P_V = matrix(2 * Sigma_V)
q_V = matrix(np.zeros(n_V))

def solve_mv_V(beta_target):
    A = matrix(np.vstack([beta_V, np.ones(n_V)]))
    b = matrix([beta_target, 1.0])
    sol = solvers.qp(P_V, q_V, None, None, A, b)
    return np.array(sol['x']).flatten()

# Solve for beta-neutral MVP
w_beta_neutral = solve_mv_V(0.0)

print("\nBeta-neutral MVP in the enlarged universe V = :")
print(pd.Series(w_beta_neutral, index=tickers_V))

print("\nBeta of beta-neutral portfolio (should be ~0):")
print(np.dot(beta_V, w_beta_neutral))

#Interpretation Part(2(c)): Optimizes risk using ALL assets (including AAPL) in V, compute the pure risk-minimizing portfolio with beta = 0
#How do strat 2b compare to 2c?
#The AAPL-hedged portfolio from Part 2(b) still carries meaningful idiosyncratic risk: 
#although its market beta is reduced, it remains volatile, experiences sizable drawdowns, and delivers a slightly negative expected return. 
#In contrast, the beta-neutral minimum-variance portfolio is almost perfectly market-neutral, with near-zero beta, extremely low volatility, 
#and negligible tail risk. Despite taking far less risk, it produces a small positive expected return. 
#In short, the AAPL hedge behaves like a higher-risk, stock-specific trade, 
#while the beta-neutral MVP is a very stable, low-risk portfolio with smoother performance.


#-----------------------------------------------------------------Part 3-----------------------------------------------------------------------
#Motivation: Comparison of the two hedging strats in 2(b) and 2(c)
#Compute the realized daily returns of the Portfolios built in questions 2(a) and 2(b)
#from period starting in January 3rd, 2025 and ending on March 30th 2025
test_start = "2025-01-03"
test_end = "2025-03-30"

returns_test = returns.loc[test_start:test_end]

print("\nTest window shape:", returns_test.shape)
print(returns_test.head())

# Strategy 1: AAPL hedged with -ω_mv(beta_AAPL) units of tickers in U
returns_U_test = returns_test[tickers_U]

R_AAPL_test = returns_test["AAPL"].values
R_hedge_test = R_AAPL_test - returns_U_test.values @ w_hedge

# Convert to Series for convenience
R_hedged_series = pd.Series(R_hedge_test, index=returns_U_test.index, name="Hedged_AAPL")

#Strategy 2: Beta-neutral MVP
tickers_V = tickers_U + ["AAPL"]
returns_V_test = returns_test[tickers_V]

R_beta_neutral_test = returns_V_test.values @ w_beta_neutral
R_beta_neutral_series = pd.Series(R_beta_neutral_test, index=returns_V_test.index, name="BetaNeutral")

results = pd.concat([R_hedged_series, R_beta_neutral_series], axis=1) #combine the two series into one dataframe
print(results.head())

cum_returns = (1 + results).cumprod() - 1 #cumulative returns: ((1+r_1)(1+r_2)...(1+r_t)) - 1

#Summary statistics
summary = pd.DataFrame({
    "Mean": results.mean(),
    "Volatility": results.std(),
    "Skewness": results.skew(),
    "Kurtosis": results.kurtosis(),
    "VaR_95": results.quantile(0.05)
})

print("\nPerformance Summary:")
print(summary)

R_SPY_test = returns_test["SPY"]

#each strat's beta relative to SPY is: beta_strat_{1,2} = Cov(Return_strat_{1,2}, Return_SPY)/Var(Return_SPY)
def compute_beta(x):
    c = np.cov(x.values, R_SPY_test.values)
    return c[0,1] / c[1,1]

beta_hedged = compute_beta(R_hedged_series)
beta_neutral = compute_beta(R_beta_neutral_series)

print("\nBetas:")
print("Hedged AAPL beta:", beta_hedged)
print("Beta-neutral portfolio beta:", beta_neutral)

cum_returns.plot(figsize=(10,5), title="Cumulative Returns") #Plot cumulative PnL
plt.show()

#Interpretation of Summary Statistics:
#Over the out-of-sample period from January 3 to March 30, 2025, the two hedging strategies behave very differently.
#The AAPL-hedged portfolio exhibits substantial volatility and meaningful drawdowns, falling more than 10% at its
#worst point before partially recovering, ultimately ending the period with a small loss.
#Its daily statistics reflect this risk: a negative average return, high volatility (~1.67% per day),
#a fat-tailed return distribution (excess kurtosis ≈ 1.68), and a large 95% daily Value-at-Risk of about –3%. 
#Although the hedge reduces systematic risk relative to holding AAPL outright, its realized beta is still around –0.34,
#indicating that estimation error and AAPL-specific shocks dominate the dynamics. 
#In contrast, the beta-neutral minimum-variance portfolio delivers a very stable return profile,
#with nearly monotonic cumulative performance, extremely low volatility, negligible tail risk, and a near-zero realized beta (–0.0026). 
#Despite its low risk, it generates a small positive mean return. 
#Overall, the AAPL-hedged strategy remains exposed to idiosyncratic AAPL fluctuations, 
#while the beta-neutral MVP behaves as a true low-risk, market-neutral portfolio with smooth, consistent performance.