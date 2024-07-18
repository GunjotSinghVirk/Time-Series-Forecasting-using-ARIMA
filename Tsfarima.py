import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# Load the dataset (you'll need to download it)
data = pd.read_csv('monthly-car-sales.csv', parse_dates=['Month'], index_col='Month')

# Split the data
train_data = data[:len(data)-12]
test_data = data[-12:]

# Fit the ARIMA model
model = ARIMA(train_data['Sales'], order=(1,1,1))
model_fit = model.fit()

# Make predictions
forecast = model_fit.forecast(steps=12)

# Evaluate the model
mse = np.mean((test_data['Sales'] - forecast)**2)
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error: {rmse:.2f}")

# Plot the results
plt.figure(figsize=(12,6))
plt.plot(train_data.index, train_data['Sales'], label='Training Data')
plt.plot(test_data.index, test_data['Sales'], label='Actual Sales')
plt.plot(test_data.index, forecast, label='Forecasted Sales')
plt.title('Monthly Car Sales Forecast')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.show()
