# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 17:09:04 2021

@author: Xiaomi
"""
# =============================================================================
# import yfinance as yf
# 
# # Get the data for the stock AAPL
# data = yf.download('AAPL','2016-01-01','2019-08-01')
# 
# # Import the plotting library
# import matplotlib.pyplot as plt
# # =============================================================================
# # matplotlib inline
# # =============================================================================
# 
# # Plot the close price of the AAPL
# data['Adj Close'].plot()
# plt.show()
# =============================================================================

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

stock_name = 'Gas'
S_eon=pd.read_excel(r'C:\Users\Xiaomi\TTF_2.xlsx')

start_date = '2016-01-01'
end_date = '2018-10-01'
pred_end_date = '2019-03-31'

plt.figure(figsize = (15, 5))
plt.plot(S_eon['Date'], S_eon['Price'])
plt.xlabel('Days')
plt.ylabel('Stock Prices, €')
plt.show()

# Parameter Assignments
So = S_eon.loc[S_eon.shape[0] - 1, "Price"]
print(So)
dt = 1 # day   # User input
n_of_wkdays = pd.date_range(start = pd.to_datetime(end_date, 
                 format = "%Y-%m-%d") + pd.Timedelta('1 days'), 
                 end = pd.to_datetime(pred_end_date, 
                 format = "%Y-%m-%d")).to_series().map(lambda x: 
                 1 if x.isoweekday() in range(1,6) else 0).sum()
T = n_of_wkdays # days  # User input -> follows from pred_end_date
N = T / dt
t = np.arange(1, int(N) + 1)
returns = (S_eon.loc[1:, 'Price'] - \
          S_eon.shift(1).loc[1:, 'Price']) / \
          S_eon.shift(1).loc[1:, 'Price']
print(returns.tolist())
mu = np.mean(returns)
sigma = np.std(returns)
scen_size = 1000 # User input
b = {str(scen): np.random.normal(0, 1, int(N)) for scen in range(1, scen_size + 1)}
W = {str(scen): b[str(scen)].cumsum() for scen in range(1, scen_size + 1)}

# Calculating drift and diffusion components
drift = (mu - 0.5 * sigma**2) * t
print(drift)
diffusion = {str(scen): sigma * W[str(scen)] for scen in range(1, scen_size + 1)}
print(diffusion)

# Making the predictions
S = np.array([So * np.exp(drift + diffusion[str(scen)]) for scen in range(1, scen_size + 1)]) 
S = np.hstack((np.array([[So] for scen in range(scen_size)]), S)) # add So to the beginning series
print(S)

# Plotting the simulations
plt.figure(figsize = (20,10))
for i in range(scen_size):
    plt.title("Daily Volatility: " + str(sigma))
    plt.plot(pd.date_range(start = S_eon["Date"].max(), 
                end = pred_end_date, freq = 'D').map(lambda x:
                x if x.isoweekday() in range(1, 6) else np.nan).dropna(), S[i, :])
    plt.ylabel('Stock Prices, €')
    plt.xlabel('Prediction Days')
plt.show()

# Dataframe format for predictions - first 10 scenarios only
Preds_df = pd.DataFrame(S.swapaxes(0, 1)[:, :1000]).set_index(
           pd.date_range(start = S_eon["Date"].max(), 
           end = pred_end_date, freq = 'D').map(lambda x:
           x if x.isoweekday() in range(1, 6) else np.nan).dropna()
           ).reset_index(drop = False)
        
print(Preds_df.head())        
Preds_df.to_excel("all.xlsx")  