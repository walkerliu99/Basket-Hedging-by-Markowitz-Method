import yfinance as yf
import pandas as pd
import numpy as np
from cvxopt import matrix, solvers
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

plt.style.use('dark_background')

#-----------------------------------------------Data Visualization Setup-------------------------------------------------------------------------------
def render_table(df, title):
    fig, ax = plt.subplots(figsize=(10, len(df) * 0.6 + 1.5))
    ax.axis('off')
    ax.set_title(title, fontsize=14, fontweight='bold', color='white', pad=20)
    table = ax.table(
        cellText=df.round(6).values,
        colLabels=df.columns,
        rowLabels=df.index,
        cellLoc='center',
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    for key, cell in table.get_celld().items():
        cell.set_edgecolor('gray')
        cell.set_text_props(color='white')
        cell.set_facecolor('#1e1e1e')
    plt.tight_layout()
    plt.show()
#---------------------------------------------------------------------------------------------------------------------

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

render_table(summary, "Performance Summary: Hedged AAPL vs Beta-Neutral")

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

fig, ax = plt.subplots(figsize=(12, 6))
cum_returns.plot(ax=ax, linewidth=1.5)
ax.set_title("Cumulative Returns", fontsize=14, fontweight='bold')
ax.set_xlabel("Date")
ax.set_ylabel("Cumulative Return")
ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))  # tick every week
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
plt.xticks(rotation=45)
plt.tight_layout()
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

#-----------------------------------------------------------------Part 4-----------------------------------------------------------------------
#4a We introduce a linear term -lambda*rho^T*w and minimize (w^T*Sigma*w - lambda*rho^T*w)
def solve_mv_with_lambda(beta_target, lam):
    P = matrix(2 * Sigma) # P = 2 Sigma
    
    q = matrix(-lam * mu_U.values) # q = -lambda * mu_U

    A = matrix(np.vstack([beta_vec, np.ones(n)]))
    b = matrix([beta_target, 1.0]) #Same constraints as previous setup.
    
    sol = solvers.qp(P, q, None, None, A, b)
    w = np.array(sol['x']).flatten()
    return w / np.sum(w)

lam_list = [i * 0.1 for i in range(0, 11)] #Let's try some lambda values, let's say from 0.0 to 1.0 with 0.1 increment.

for lam in lam_list:
    w_lam = solve_mv_with_lambda(beta_AAPL, lam)
    print(f"lambda={lam}, expected daily return={mu_U.values @ w_lam}") #the larger the scaler lambda, the larger the expected daily return required make it worthy of the risk.    

returns_U_test = returns_test[tickers_U]
R_AAPL_test = returns_test["AAPL"].values

results_lam = {} 

#we fill a dictionary of results for different lambda
for lam in lam_list:
    lam_str = format(lam, ".1f")
    w_lam = solve_mv_with_lambda(beta_AAPL, lam)
    R_lam = R_AAPL_test - returns_U_test.values @ w_lam
    results_lam[lam] = pd.Series(R_lam, index=returns_U_test.index, name=f"Hedge_lambda_{lam_str}") 

#we can visualize the PnL graph from 1/3/25 to 3/1/25
df_lam = pd.concat(results_lam.values(), axis=1)
cum_lam = (1 + df_lam).cumprod() - 1

fig, ax = plt.subplots(figsize=(12, 6))
cum_lam.plot(ax=ax, linewidth=1.5)
ax.set_title("Cumulative Returns for Different λ Values", fontsize=14, fontweight='bold')
ax.set_xlabel("Date")
ax.set_ylabel("Cumulative Return")
ax.set_ylim(-0.4, 0.4)
ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
plt.xticks(rotation=45)
plt.legend(loc='upper left', fontsize=8)
plt.tight_layout()
plt.show()


#Comment: As soon as we introduce a positive λ, the optimizer produces leveraged portfolios, leading to huge swings in cumulative return.(the larger the lambda, the bigger the swings)
#We can compare different lambda's PnL to the portfolio of lambda = 0
col0 = "Hedge_lambda_0.0"
final_0 = cum_lam[col0].iloc[-1]

print("Final return for λ = 0.0:", final_0)
print("\nLambdas outperforming λ = 0.0:")

for lam in lam_list:
    lam_str = format(lam, ".1f")
    col = f"Hedge_lambda_{lam_str}"
    
    if lam == 0:
        continue
        
    final_lam = cum_lam[col].iloc[-1]
    
    if final_lam > final_0:
        print(f"λ = {lam_str} --> outperforms (final return = {final_lam:.4f})") #see which lambda parameter perform the best

#Summary: We can see from the output that lambda = 0.3 seem to be giving the best PnL on 3/1/25 with 2.57% return.

#4b Dynamic Hedging with 60-day rolling hedge, weekly rebalance
window = 60
test_dates = returns_test.index

dynamic_returns = []
current_w = None

for i, date in enumerate(test_dates):
    if i % 5 == 0 or current_w is None:
        idx = returns.index.get_loc(date)
        start_idx = max(0, idx - window)
        past = returns.iloc[start_idx:idx]

        window_U = past[tickers_U]
        window_SPY = past["SPY"]

        Sigma_temp = window_U.cov().values

        beta_vec_temp = []
        for t in tickers_U:
            cov_ts = np.cov(past[t].values, window_SPY.values)
            beta_vec_temp.append(cov_ts[0,1] / cov_ts[1,1])
        beta_vec_temp = np.array(beta_vec_temp)

        cov_aapl_spy_temp = np.cov(past["AAPL"].values, window_SPY.values)
        beta_AAPL_temp = cov_aapl_spy_temp[0,1] / cov_aapl_spy_temp[1,1]

        P_temp = matrix(2 * Sigma_temp)
        q_temp = matrix(np.zeros(len(tickers_U)))
        A_temp = matrix(np.vstack([beta_vec_temp, np.ones(len(tickers_U))]))
        b_temp = matrix([beta_AAPL_temp, 1.0])

        sol_temp = solvers.qp(P_temp, q_temp, None, None, A_temp, b_temp)
        current_w = np.array(sol_temp['x']).flatten()

    R_AAPL_t = returns.loc[date, "AAPL"]
    R_U_t = returns.loc[date, tickers_U].values
    R_dyn_t = R_AAPL_t - R_U_t @ current_w
    dynamic_returns.append(R_dyn_t)

dynamic_hedge_series = pd.Series(dynamic_returns, index=test_dates, name="DynamicHedge")

results_all = pd.concat([results, dynamic_hedge_series], axis=1)

cum_all = (1 + results_all).cumprod() - 1

fig, ax = plt.subplots(figsize=(12, 6))
cum_all.plot(ax=ax, linewidth=1.5)
ax.set_title("Cumulative Returns: Static Hedge vs Beta-Neutral vs Dynamic Hedge", fontsize=14, fontweight='bold')
ax.set_xlabel("Date")
ax.set_ylabel("Cumulative Return")
ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
plt.xticks(rotation=45)
plt.legend(loc='best', fontsize=10)
plt.tight_layout()
plt.show()


summary_all = pd.DataFrame({
    "Mean": results_all.mean(),
    "Volatility": results_all.std(),
    "Skewness": results_all.skew(),
    "Kurtosis": results_all.kurtosis(),
    "VaR_95": results_all.quantile(0.05)
})

render_table(summary_all, "Performance Summary: Static Hedge vs Dynamic Hedge vs Beta-Neutral")


beta_dynamic = compute_beta(dynamic_hedge_series)
print("\nDynamic hedge beta vs SPY:", beta_dynamic)
