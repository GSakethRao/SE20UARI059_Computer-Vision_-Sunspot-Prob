#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.datasets import co2
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Obtain CO2 dataset from statsmodels
dataset = co2.load_pandas().data
dataset.index = pd.date_range(start='1958-03-29', periods=len(dataset), freq='W-SAT')
co2_levels = dataset['co2']

# Visualizing CO2 levels over time
plt.figure(figsize=(12, 6))
plt.plot(co2_levels)
plt.title("CO2 Levels Through the Years")
plt.xlabel("Year")
plt.ylabel("CO2 Level")
plt.show()

# Analyzing AutoCorrelation and Partial AutoCorrelation
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plot_acf(co2_levels, lags=40)
plot_pacf(co2_levels, lags=40)
plt.show()

# SARIMA parameters
p, d, q = 1, 1, 1
P, D, Q, S = 1, 1, 1, 12

# Constructing and fitting the model
model = SARIMAX(co2_levels, order=(p, d, q), seasonal_order=(P, D, Q, S))
model_results = model.fit()

# Dividing dataset into training and testing sets
partition = int(len(co2_levels) * 0.8)
training_data, testing_data = co2_levels[:partition], co2_levels[partition:]

# Training SARIMA model on the training data
trained_model = SARIMAX(training_data, order=(p, d, q), seasonal_order=(P, D, Q, S))
trained_results = trained_model.fit()

# Making forecast
forecast_start = len(training_data)
forecast_end = len(training_data) + len(testing_data) - 1
forecasted_values = trained_results.predict(start=forecast_start, end=forecast_end, dynamic=False)

# Plotting original, testing and forecasted data
plt.figure(figsize=(12, 6))
plt.plot(training_data, label='Training Data')
plt.plot(testing_data, label='Testing Data')
plt.plot(forecasted_values, label='Forecasted Data')
plt.title("Forecasting CO2 Levels")
plt.legend()
plt.show()

# Evaluating the model's performance
mae_val = mean_absolute_error(testing_data, forecasted_values)
mse_val = mean_squared_error(testing_data, forecasted_values)
rmse_val = np.sqrt(mse_val)

print(f"MAE: {mae_val}")
print(f"MSE: {mse_val}")
print(f"RMSE: {rmse_val}")


# In[ ]:




