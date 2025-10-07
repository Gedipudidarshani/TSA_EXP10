# Exp.no: 10   IMPLEMENTATION OF SARIMA MODEL
### Date: 07-10-2025
#### NAME:GEDIPUDI DARSHANI
#### REGISTER NUMBER:212223230062

### AIM:
To implement SARIMA model using python.
### ALGORITHM:
1. Explore the dataset
2. Check for stationarity of time series
3. Determine SARIMA models parameters p, q
4. Fit the SARIMA model
5. Make time series predictions and Auto-fit the SARIMA model
6. Evaluate model predictions
### PROGRAM:
```
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

# -----------------------------
# 1️⃣ Load and preprocess data
# -----------------------------
data = pd.read_csv('/content/INDIA VIX_minute.csv')

# Convert 'date' to datetime and set as index
data['date'] = pd.to_datetime(data['date'])
data.set_index('date', inplace=True)

# If data is at minute level, aggregate monthly (or daily if you prefer)
# You can also try 'D' for daily or 'W' for weekly
data_monthly = data['open'].resample('MS').mean()

# Visualize the time series
plt.figure(figsize=(10, 6))
plt.plot(data_monthly, label='Open (Monthly Avg)', color='blue')
plt.title('INDIA VIX Monthly Open Time Series')
plt.xlabel('Date')
plt.ylabel('Open Value')
plt.legend()
plt.grid()
plt.show()

# -----------------------------
# 2️⃣ Stationarity Test (ADF)
# -----------------------------
def test_stationarity(timeseries):
    result = adfuller(timeseries.dropna())
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    for key, value in result[4].items():
        print(f'Critical Value {key}: {value}')
    if result[1] < 0.05:
        print("✅ The time series is stationary.\n")
    else:
        print("❌ The time series is not stationary.\n")

print("Before Differencing:")
test_stationarity(data_monthly)

# Difference if needed
data_diff = data_monthly.diff().dropna()
print("After Differencing:")
test_stationarity(data_diff)

# -----------------------------
# 3️⃣ Build SARIMA Model
# -----------------------------
# You can adjust (p,d,q) and (P,D,Q,m)
p, d, q = 1, 1, 1
P, D, Q, m = 1, 1, 1, 12  # m=12 for monthly seasonality

model = SARIMAX(data_monthly,
                order=(p, d, q),
                seasonal_order=(P, D, Q, m),
                enforce_stationarity=False,
                enforce_invertibility=False)

sarima_fit = model.fit(disp=False)
print(sarima_fit.summary())

# -----------------------------
# 4️⃣ Forecasting
# -----------------------------
forecast_steps = 12  # forecast for next 12 months
forecast = sarima_fit.get_forecast(steps=forecast_steps)
forecast_ci = forecast.conf_int()

# Align indexes (no timezone)
data_monthly.index = pd.to_datetime(data_monthly.index).tz_localize(None)
forecast.predicted_mean.index = pd.to_datetime(forecast.predicted_mean.index).tz_localize(None)
forecast_ci.index = pd.to_datetime(forecast_ci.index).tz_localize(None)

# -----------------------------
# 5️⃣ Visualization
# -----------------------------
plt.figure(figsize=(12, 6))
plt.plot(data_monthly, label='Historical Data', color='blue')
plt.plot(forecast.predicted_mean, label='Forecast', color='red')
plt.fill_between(forecast_ci.index,
                 forecast_ci.iloc[:, 0],
                 forecast_ci.iloc[:, 1], color='pink', alpha=0.3)
plt.title('SARIMA Forecast of INDIA VIX (Open Values)')
plt.xlabel('Date')
plt.ylabel('Open')
plt.legend()
plt.grid()
plt.show()

# -----------------------------
# 6️⃣ Model Evaluation
# -----------------------------
# Use the last 12 months as test data
test_data = data_monthly[-forecast_steps:]
pred_data = forecast.predicted_mean[:len(test_data)]

mae = mean_absolute_error(test_data, pred_data)
print(f"Mean Absolute Error (MAE): {mae:.4f}")

```
### OUTPUT:

<img width="957" height="611" alt="image" src="https://github.com/user-attachments/assets/2c19c3ec-b416-46a4-83b8-782045d22df9" />

<img width="527" height="324" alt="image" src="https://github.com/user-attachments/assets/3b4b418a-d5a2-4a36-a794-25dd67e82208" />

<img width="823" height="489" alt="image" src="https://github.com/user-attachments/assets/9186b46f-bc83-4905-bd63-7d998963d506" />

<img width="1120" height="637" alt="image" src="https://github.com/user-attachments/assets/edd63994-a9b7-459c-9b19-cda364a400bc" />


### RESULT:
Thus the program run successfully based on the SARIMA model.
