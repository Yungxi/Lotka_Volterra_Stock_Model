"""
Lotka-Volterra Competition Model for Stock Prices

Models two competing company stock prices as competing species.
The competition variant of Lotka-Volterra equations:
    dN1/dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)
    dN2/dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)

Where:
    N1, N2: Stock prices (normalized)
    r1, r2: Intrinsic growth rates
    K1, K2: Carrying capacities
    alpha12: Competition effect of company 2 on company 1
    alpha21: Competition effect of company 1 on company 2
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import minimize
import yfinance as yf
from datetime import datetime, timedelta


def lotka_volterra_competition(y, t, r1, r2, K1, K2, alpha12, alpha21, K_growth_rate=0.0):
    """
    Lotka-Volterra competition equations for two species.
    Carrying capacity grows over time to reflect market growth.

    K(t) = K0 * (1 + K_growth_rate)^(t/252)
    where 252 = trading days per year
    """
    N1, N2 = y

    # Time-varying carrying capacity (grows with market)
    growth_factor = (1 + K_growth_rate) ** (t / 252)
    K1_t = K1 * growth_factor
    K2_t = K2 * growth_factor

    dN1dt = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1_t)
    dN2dt = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2_t)
    return [dN1dt, dN2dt]


def fetch_stock_data(ticker1, ticker2, start_date, end_date):
    """
    Fetch historical stock data for two companies.
    """
    stock1 = yf.download(ticker1, start=start_date, end=end_date, progress=False)
    stock2 = yf.download(ticker2, start=start_date, end=end_date, progress=False)

    # Handle different yfinance column formats
    # Newer versions may use MultiIndex or different column names
    def get_close_prices(df):
        if 'Adj Close' in df.columns:
            return df['Adj Close'].values.flatten()
        elif 'Close' in df.columns:
            return df['Close'].values.flatten()
        elif isinstance(df.columns, pd.MultiIndex):
            # MultiIndex columns - try to get Close or Adj Close
            if 'Adj Close' in df.columns.get_level_values(0):
                return df['Adj Close'].values.flatten()
            elif 'Close' in df.columns.get_level_values(0):
                return df['Close'].values.flatten()
        # Fallback: use the first price-like column
        return df.iloc[:, 0].values.flatten()

    prices1 = get_close_prices(stock1)
    prices2 = get_close_prices(stock2)

    # Align data (use minimum length)
    min_len = min(len(prices1), len(prices2))
    prices1 = prices1[:min_len]
    prices2 = prices2[:min_len]

    return prices1, prices2


def normalize_prices(prices1, prices2):
    """
    Normalize prices to range [0, 1] for better model fitting.
    """
    all_prices = np.concatenate([prices1, prices2])
    min_val, max_val = all_prices.min(), all_prices.max()

    norm1 = (prices1 - min_val) / (max_val - min_val)
    norm2 = (prices2 - min_val) / (max_val - min_val)

    return norm1, norm2, min_val, max_val


def objective_function(params, t, actual1, actual2, K_growth_rate=0.0):
    """
    Objective function for parameter optimization.
    Minimizes the sum of squared errors between predicted and actual prices.
    """
    r1, r2, K1, K2, alpha12, alpha21 = params

    # Ensure positive parameters
    if any(p <= 0 for p in params):
        return 1e10

    y0 = [actual1[0], actual2[0]]

    try:
        solution = odeint(lotka_volterra_competition, y0, t,
                         args=(r1, r2, K1, K2, alpha12, alpha21, K_growth_rate))
        pred1, pred2 = solution[:, 0], solution[:, 1]

        # Sum of squared errors
        error = np.sum((pred1 - actual1)**2 + (pred2 - actual2)**2)
        return error
    except:
        return 1e10


def fit_model(t, actual1, actual2, K_growth_rate=0.0):
    """
    Fit Lotka-Volterra parameters to actual stock data.
    """
    # Initial parameter guesses
    initial_params = [0.1, 0.1, 1.5, 1.5, 0.5, 0.5]

    # Parameter bounds
    bounds = [(0.001, 1), (0.001, 1), (0.5, 3), (0.5, 3), (0.01, 2), (0.01, 2)]

    result = minimize(
        objective_function,
        initial_params,
        args=(t, actual1, actual2, K_growth_rate),
        method='L-BFGS-B',
        bounds=bounds
    )

    return result.x


def predict(params, t, y0, K_growth_rate=0.0):
    """
    Generate predictions using fitted parameters (deterministic).
    """
    r1, r2, K1, K2, alpha12, alpha21 = params
    solution = odeint(lotka_volterra_competition, y0, t,
                     args=(r1, r2, K1, K2, alpha12, alpha21, K_growth_rate))
    return solution[:, 0], solution[:, 1]


def estimate_volatility(actual, predicted):
    """
    Estimate volatility (sigma) from residuals between actual and predicted.
    """
    residuals = actual - predicted
    return np.std(residuals)


def estimate_correlation(actual1, actual2):
    """
    Estimate correlation between two stock return series.
    """
    returns1 = np.diff(actual1) / (actual1[:-1] + 1e-8)
    returns2 = np.diff(actual2) / (actual2[:-1] + 1e-8)
    return np.corrcoef(returns1, returns2)[0, 1]


def predict_stochastic(params, t, y0, sigma1, sigma2, n_simulations=500, rho=0.0, K_growth_rate=0.0):
    """
    Generate stochastic predictions using Euler-Maruyama method.
    Runs Monte Carlo simulations to get confidence intervals.

    Uses correlated Brownian motions:
    dN1 = drift1 * dt + sigma1 * N1 * dW1
    dN2 = drift2 * dt + sigma2 * N2 * dW2

    Where dW2 = rho * dW1 + sqrt(1-rho^2) * dZ (dZ independent of dW1)
    """
    r1, r2, K1, K2, alpha12, alpha21 = params
    dt = 1.0  # 1 day time step
    n_steps = len(t)

    # Store all simulation paths
    all_paths1 = np.zeros((n_simulations, n_steps))
    all_paths2 = np.zeros((n_simulations, n_steps))

    for sim in range(n_simulations):
        N1 = np.zeros(n_steps)
        N2 = np.zeros(n_steps)
        N1[0], N2[0] = y0

        for i in range(1, n_steps):
            # Time-varying carrying capacity
            growth_factor = (1 + K_growth_rate) ** (t[i] / 252)
            K1_t = K1 * growth_factor
            K2_t = K2 * growth_factor

            # Deterministic drift (Lotka-Volterra with growing K)
            drift1 = r1 * N1[i-1] * (1 - (N1[i-1] + alpha12 * N2[i-1]) / K1_t)
            drift2 = r2 * N2[i-1] * (1 - (N2[i-1] + alpha21 * N1[i-1]) / K2_t)

            # Correlated Brownian motions
            # dW1 is independent, dW2 is correlated with dW1
            dW1 = np.random.normal(0, np.sqrt(dt))
            dZ = np.random.normal(0, np.sqrt(dt))  # Independent noise
            dW2 = rho * dW1 + np.sqrt(1 - rho**2) * dZ  # Correlated noise

            # Euler-Maruyama update
            N1[i] = N1[i-1] + drift1 * dt + sigma1 * N1[i-1] * dW1
            N2[i] = N2[i-1] + drift2 * dt + sigma2 * N2[i-1] * dW2

            # Ensure non-negative prices
            N1[i] = max(N1[i], 0.001)
            N2[i] = max(N2[i], 0.001)

        all_paths1[sim] = N1
        all_paths2[sim] = N2

    # Calculate statistics across simulations
    mean1 = np.mean(all_paths1, axis=0)
    mean2 = np.mean(all_paths2, axis=0)

    # Confidence intervals (5th and 95th percentile for 90% CI)
    lower1 = np.percentile(all_paths1, 5, axis=0)
    upper1 = np.percentile(all_paths1, 95, axis=0)
    lower2 = np.percentile(all_paths2, 5, axis=0)
    upper2 = np.percentile(all_paths2, 95, axis=0)

    return mean1, mean2, lower1, upper1, lower2, upper2


def plot_results_with_forecast(t, actual1, actual2, pred1, pred2, train_len, ticker1, ticker2,
                                stochastic=False, lower1=None, upper1=None, lower2=None, upper2=None):
    """
    Plot actual vs predicted stock prices with train/test split visualization.
    Optionally shows confidence intervals for stochastic predictions.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    t_test = t[train_len:]

    # Plot 1: Both stocks - Training vs Forecast
    ax1 = axes[0, 0]
    ax1.axvspan(0, train_len, alpha=0.1, color='green', label='Training Period')
    ax1.axvspan(train_len, len(t), alpha=0.1, color='orange', label='Forecast Period')
    ax1.axvline(x=train_len, color='black', linestyle='--', linewidth=2, alpha=0.7)
    ax1.plot(t, actual1, 'b-', label=f'{ticker1} Actual', linewidth=2)
    ax1.plot(t, actual2, 'r-', label=f'{ticker2} Actual', linewidth=2)
    ax1.plot(t, pred1, 'b--', label=f'{ticker1} Predicted', linewidth=2, alpha=0.7)
    ax1.plot(t, pred2, 'r--', label=f'{ticker2} Predicted', linewidth=2, alpha=0.7)
    if stochastic and lower1 is not None:
        ax1.fill_between(t, lower1, upper1, alpha=0.2, color='blue')
        ax1.fill_between(t, lower2, upper2, alpha=0.2, color='red')
    ax1.set_xlabel('Time (days)')
    ax1.set_ylabel('Normalized Price')
    title = 'Stochastic Lotka-Volterra Forecast' if stochastic else 'Lotka-Volterra Forecast'
    ax1.set_title(f'{title}: Train/Test Split')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Company 1 - with confidence interval
    ax2 = axes[0, 1]
    ax2.axvspan(0, train_len, alpha=0.1, color='green')
    ax2.axvspan(train_len, len(t), alpha=0.1, color='orange')
    ax2.axvline(x=train_len, color='black', linestyle='--', linewidth=2, alpha=0.7, label='Forecast Start')
    ax2.plot(t, actual1, 'b-', label='Actual', linewidth=2)
    ax2.plot(t, pred1, 'b--', label='Predicted (Mean)' if stochastic else 'Predicted', linewidth=2, alpha=0.7)
    if stochastic and lower1 is not None:
        ax2.fill_between(t, lower1, upper1, alpha=0.3, color='blue', label='90% CI')
    else:
        ax2.fill_between(t_test, actual1[train_len:], pred1[train_len:], alpha=0.4, color='orange', label='Forecast Error')
    ax2.set_xlabel('Time (days)')
    ax2.set_ylabel('Normalized Price')
    ax2.set_title(f'{ticker1} - {"Stochastic " if stochastic else ""}Forecast')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Company 2 - with confidence interval
    ax3 = axes[1, 0]
    ax3.axvspan(0, train_len, alpha=0.1, color='green')
    ax3.axvspan(train_len, len(t), alpha=0.1, color='orange')
    ax3.axvline(x=train_len, color='black', linestyle='--', linewidth=2, alpha=0.7, label='Forecast Start')
    ax3.plot(t, actual2, 'r-', label='Actual', linewidth=2)
    ax3.plot(t, pred2, 'r--', label='Predicted (Mean)' if stochastic else 'Predicted', linewidth=2, alpha=0.7)
    if stochastic and lower2 is not None:
        ax3.fill_between(t, lower2, upper2, alpha=0.3, color='red', label='90% CI')
    else:
        ax3.fill_between(t_test, actual2[train_len:], pred2[train_len:], alpha=0.4, color='orange', label='Forecast Error')
    ax3.set_xlabel('Time (days)')
    ax3.set_ylabel('Normalized Price')
    ax3.set_title(f'{ticker2} - {"Stochastic " if stochastic else ""}Forecast')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Forecast accuracy / CI width over time
    ax4 = axes[1, 1]
    if stochastic and lower1 is not None:
        # Show CI width growing over forecast horizon
        ci_width1 = (upper1 - lower1)[train_len:]
        ci_width2 = (upper2 - lower2)[train_len:]
        days_ahead = np.arange(1, len(ci_width1) + 1)
        ax4.plot(days_ahead, ci_width1, 'b-', label=f'{ticker1} CI Width', linewidth=2)
        ax4.plot(days_ahead, ci_width2, 'r-', label=f'{ticker2} CI Width', linewidth=2)
        ax4.set_ylabel('Confidence Interval Width')
        ax4.set_title('Uncertainty Growth Over Forecast Horizon')
    else:
        forecast_error1 = np.abs(actual1[train_len:] - pred1[train_len:])
        forecast_error2 = np.abs(actual2[train_len:] - pred2[train_len:])
        days_ahead = np.arange(1, len(forecast_error1) + 1)
        ax4.plot(days_ahead, forecast_error1, 'b-', label=f'{ticker1}', linewidth=2)
        ax4.plot(days_ahead, forecast_error2, 'r-', label=f'{ticker2}', linewidth=2)
        ax4.set_ylabel('Absolute Error')
        ax4.set_title('Forecast Error vs Horizon')
    ax4.set_xlabel('Days Ahead (Forecast Horizon)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    filename = 'lotka_volterra_stochastic.png' if stochastic else 'lotka_volterra_forecast.png'
    plt.savefig(filename, dpi=150)
    plt.show()
    return filename


def main():
    # ============== CONFIGURATION ==============
    ticker1 = "UBER"
    ticker2 = "LYFT"
    train_ratio = 0.3           # Use 80% for training, 20% for testing
    use_stochastic = True       # Set to False for deterministic predictions
    n_simulations = 500         # Number of Monte Carlo simulations (if stochastic)
    use_correlation = True      # Use correlated noise between stocks
    K_growth_rate = 0.01        # Carrying capacity growth rate (1% per year)
    # ===========================================

    # Fetch 1 year of data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)

    print(f"Fetching stock data for {ticker1} and {ticker2}...")
    prices1, prices2 = fetch_stock_data(
        ticker1, ticker2,
        start_date.strftime('%Y-%m-%d'),
        end_date.strftime('%Y-%m-%d')
    )

    print(f"Total data points: {len(prices1)}")
    print(f"Mode: {'Stochastic' if use_stochastic else 'Deterministic'}")

    # Normalize prices
    norm1, norm2, min_val, max_val = normalize_prices(prices1, prices2)

    # Time array
    t = np.arange(len(norm1))

    # Train/test split
    train_len = int(len(norm1) * train_ratio)
    test_len = len(norm1) - train_len
    print(f"Training on: {train_len} days")
    print(f"Forecasting: {test_len} days")

    # Fit model ONLY on training data
    t_train = t[:train_len]
    norm1_train = norm1[:train_len]
    norm2_train = norm2[:train_len]

    print("\nFitting Lotka-Volterra model on training data...")
    print(f"Carrying capacity growth rate: {K_growth_rate*100:.1f}% per year")
    params = fit_model(t_train, norm1_train, norm2_train, K_growth_rate)

    r1, r2, K1, K2, alpha12, alpha21 = params
    print("\nFitted Parameters:")
    print(f"  r1 (growth rate {ticker1}): {r1:.4f}")
    print(f"  r2 (growth rate {ticker2}): {r2:.4f}")
    print(f"  K1 (carrying capacity {ticker1}): {K1:.4f}")
    print(f"  K2 (carrying capacity {ticker2}): {K2:.4f}")
    print(f"  alpha12 (effect of {ticker2} on {ticker1}): {alpha12:.4f}")
    print(f"  alpha21 (effect of {ticker1} on {ticker2}): {alpha21:.4f}")

    # Generate predictions
    y0 = [norm1[0], norm2[0]]

    # First get deterministic predictions for volatility estimation
    pred1_det, pred2_det = predict(params, t_train, y0, K_growth_rate)

    if use_stochastic:
        # Estimate volatility from training residuals
        sigma1 = estimate_volatility(norm1_train, pred1_det)
        sigma2 = estimate_volatility(norm2_train, pred2_det)
        print(f"\nEstimated Volatility:")
        print(f"  {ticker1} sigma: {sigma1:.4f}")
        print(f"  {ticker2} sigma: {sigma2:.4f}")

        # Estimate correlation between stocks
        if use_correlation:
            rho = estimate_correlation(norm1_train, norm2_train)
            print(f"\nEstimated Correlation:")
            print(f"  rho ({ticker1}, {ticker2}): {rho:.4f}")
        else:
            rho = 0.0

        print(f"\nRunning {n_simulations} Monte Carlo simulations...")
        pred1, pred2, lower1, upper1, lower2, upper2 = predict_stochastic(
            params, t, y0, sigma1, sigma2, n_simulations, rho=rho, K_growth_rate=K_growth_rate
        )
    else:
        pred1, pred2 = predict(params, t, y0, K_growth_rate)
        lower1, upper1, lower2, upper2 = None, None, None, None

    # Calculate R-squared for training period
    ss_res1_train = np.sum((norm1[:train_len] - pred1[:train_len])**2)
    ss_tot1_train = np.sum((norm1[:train_len] - np.mean(norm1[:train_len]))**2)
    r2_train1 = 1 - (ss_res1_train / ss_tot1_train)

    ss_res2_train = np.sum((norm2[:train_len] - pred2[:train_len])**2)
    ss_tot2_train = np.sum((norm2[:train_len] - np.mean(norm2[:train_len]))**2)
    r2_train2 = 1 - (ss_res2_train / ss_tot2_train)

    # Calculate R-squared for test/forecast period
    ss_res1_test = np.sum((norm1[train_len:] - pred1[train_len:])**2)
    ss_tot1_test = np.sum((norm1[train_len:] - np.mean(norm1[train_len:]))**2)
    r2_test1 = 1 - (ss_res1_test / ss_tot1_test)

    ss_res2_test = np.sum((norm2[train_len:] - pred2[train_len:])**2)
    ss_tot2_test = np.sum((norm2[train_len:] - np.mean(norm2[train_len:]))**2)
    r2_test2 = 1 - (ss_res2_test / ss_tot2_test)

    # Calculate MAPE for forecast
    mape1 = np.mean(np.abs((norm1[train_len:] - pred1[train_len:]) / (norm1[train_len:] + 1e-8))) * 100
    mape2 = np.mean(np.abs((norm2[train_len:] - pred2[train_len:]) / (norm2[train_len:] + 1e-8))) * 100

    print(f"\n--- Training Performance (R-squared) ---")
    print(f"  {ticker1}: {r2_train1:.4f}")
    print(f"  {ticker2}: {r2_train2:.4f}")

    print(f"\n--- Forecast Performance (R-squared) ---")
    print(f"  {ticker1}: {r2_test1:.4f}")
    print(f"  {ticker2}: {r2_test2:.4f}")

    print(f"\n--- Forecast MAPE (Mean Absolute % Error) ---")
    print(f"  {ticker1}: {mape1:.2f}%")
    print(f"  {ticker2}: {mape2:.2f}%")

    if use_stochastic:
        # Check if actual values fall within confidence intervals
        in_ci1 = np.mean((norm1[train_len:] >= lower1[train_len:]) & (norm1[train_len:] <= upper1[train_len:])) * 100
        in_ci2 = np.mean((norm2[train_len:] >= lower2[train_len:]) & (norm2[train_len:] <= upper2[train_len:])) * 100
        print(f"\n--- Confidence Interval Coverage (90% CI) ---")
        print(f"  {ticker1}: {in_ci1:.1f}% of actuals within CI")
        print(f"  {ticker2}: {in_ci2:.1f}% of actuals within CI")

    # Plot results
    print("\nGenerating plots...")
    filename = plot_results_with_forecast(
        t, norm1, norm2, pred1, pred2, train_len, ticker1, ticker2,
        stochastic=use_stochastic, lower1=lower1, upper1=upper1, lower2=lower2, upper2=upper2
    )
    print(f"Plot saved as '{filename}'")


if __name__ == "__main__":
    main()
