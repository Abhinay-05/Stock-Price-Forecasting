import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Ask user for input
stock = input("Enter Stock symbol (ex: TCS.NS, INFY.NS, RELIANCE.NS): ")

# downloading stock data
from datetime import datetime

end_date = datetime.today().strftime('%Y-%m-%d')
data = yf.download(stock, start='2020-01-01', end=end_date)

# if no data is added
if data.empty:
    print("Invalid stock symbol.")
    exit()

print("\nDataset Info:")
print(data.info())

print("\nSummary Statistics:")
print(data.describe())

# fix column structure
# because stock has 2 column titles
data.columns = data.columns.get_level_values(0)

# Create prediction column
data['Prediction'] = data['Close'].shift(-10)

# prepare Machine Learning dataset
features = ['Open','Close','High','Low','Volume']
x=data[features][:-10]
y=data['Prediction'][:-10]

# spliting data
# setting 20% for testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# train model
model = LinearRegression()
model.fit(x_train, y_train)

# test model
predictions = model.predict(x_test)

error = mean_squared_error(y_test, predictions)
print("Model's error:", error)

rmse = np.sqrt(error)
print("Root Mean Squared Error:", rmse)


# Predict next 10 days
future_x = data[features].tail(10)
future_predictions = model.predict(future_x)
print('Future predictions:', future_predictions)


# plotting the forecast in graph
future_dates = pd.date_range( start=data.index[-1], periods=11, freq='B')[1:]
plt.style.use("seaborn-v0_8")
plt.figure(figsize=(14,7))
plt.plot(data['Close'].tail(100), label="Historical Price")
plt.plot(future_dates, future_predictions, linestyle="dashed", label="Forecast")
plt.title(f"{stock} Stock Forecast")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid('true')
plt.show()