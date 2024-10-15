# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 11:37:30 2024

@author: hp
"""

# Import necessary libraries
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from datetime import datetime

# Download stock data
ticker = 'AAPL'  # Apple Inc. as an example
start_date = '2021-01-01'
end_date = datetime.today().strftime('%Y-%m-%d')

data = yf.download(ticker, start=start_date, end=end_date)

# Display the first few rows of the data
print(data.head())

# Calculate daily returns
data['Returns'] = data['Close'].pct_change()

# Prepare data for linear regression
data['Days'] = (data.index - data.index[0]).days
X = data[['Days']]
y = data['Close'].dropna()

# Fit linear regression model
model = LinearRegression()
model.fit(X[:-1], y[:-1])  # Exclude the last value for training

# Predict the next 30 days
future_days = np.array([(data.index[-1] + pd.Timedelta(days=i) - data.index[0]).days for i in range(1, 31)]).reshape(-1, 1)
predicted_prices = model.predict(future_days)

# Create future dates for predictions
future_dates = [data.index[-1] + pd.Timedelta(days=i) for i in range(1, 31)]
predicted_prices_df = pd.DataFrame(predicted_prices, index=future_dates, columns=['Predicted Price'])

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(data['Close'], label='Historical Closing Price', color='blue')
plt.plot(predicted_prices_df, label='Predicted Prices', color='green', linestyle='--', marker='o')
plt.title(f'{ticker} Historical Closing Prices and Predicted Prices')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.legend()
plt.grid()
plt.show()


# Portfolio Optimization Example
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']  # Example stock tickers
data = yf.download(tickers, start=start_date, end=end_date)['Close']

# Calculate daily returns
returns = data.pct_change().dropna()

# Define portfolio optimization function
def portfolio_optimization(returns):
    num_portfolios = 10000
    results = np.zeros((3, num_portfolios))
    
    for i in range(num_portfolios):
        weights = np.random.random(len(returns.columns))
        weights /= np.sum(weights)  # Normalize to 1
        
        portfolio_return = np.sum(weights * returns.mean()) * 252  # Annualized return
        portfolio_stddev = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))  # Annualized risk
        
        results[0, i] = portfolio_return
        results[1, i] = portfolio_stddev
        results[2, i] = results[0, i] / results[1, i]  # Sharpe ratio
    
    return results

results = portfolio_optimization(returns)

# Convert results to a DataFrame for visualization
results_df = pd.DataFrame(results.T, columns=['Returns', 'Risk', 'Sharpe Ratio'])

# Plot the results
plt.figure(figsize=(12, 6))
plt.scatter(results_df['Risk'], results_df['Returns'], c=results_df['Sharpe Ratio'], cmap='viridis')
plt.title('Portfolio Optimization: Risk vs. Return')
plt.xlabel('Risk (Standard Deviation)')
plt.ylabel('Expected Return')
plt.colorbar(label='Sharpe Ratio')
plt.grid()
plt.show()


# Calculate Value at Risk (VaR)
def calculate_var(returns, confidence_level=0.95):
    return np.percentile(returns, 100 * (1 - confidence_level))

# Calculate VaR for each stock
var_results = returns.aggregate(calculate_var)
print("Value at Risk (VaR) at 95% confidence level:")
print(var_results)
