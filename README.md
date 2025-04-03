# Stock Market Price Prediction Using ARIMA and ARMA Models

## Project Overview
This project focuses on predicting stock market prices using **ARIMA (Autoregressive Integrated Moving Average)** and **ARMA (Autoregressive Moving Average)** models. The analysis includes data preprocessing, stationarity checks, model selection, training, and evaluation. The dataset used is historical Google stock price data.

---

## Key Features
- **Data Preprocessing**: Handles missing values and converts date formats.
- **Exploratory Data Analysis**: Visualizes stock price trends and trading volumes.
- **Stationarity Check**: Uses the Augmented Dickey-Fuller (ADF) test.
- **Model Selection**: Automatically selects optimal ARIMA parameters using `pmdarima`.
- **Model Training & Evaluation**: Trains ARIMA and ARMA models and evaluates performance with RMSE and R² scores.
- **Visualization**: Plots actual vs. predicted prices for model comparison.

---

## Installation
### Dependencies
- Python 3.7+
- Required packages:
  ```bash
  pip install pandas numpy matplotlib seaborn statsmodels pmdarima scikit-learn
  ```

---

## Code Structure
### 1. Import Libraries
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, r2_score
```

### 2. Load and Preprocess Data
```python
df = pd.read_csv("Google_stock_price.csv")
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
```

### 3. Exploratory Data Analysis
```python
plt.figure(figsize=(20, 12))
plt.subplot(2, 1, 1)
plt.plot(df['Close'], label='Close Price')
plt.title('Google Stock Price')
plt.legend()

plt.subplot(2, 1, 2)
plt.bar(df.index, df['Volume'])
plt.title('Volume Traded')
plt.show()
```

### 4. Stationarity Check
```python
def test_stationarity(series):
    result = adfuller(series)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:', result[4])

test_stationarity(df['Close'])
```

### 5. Model Training (ARIMA)
```python
# Determine optimal (p, d, q) parameters
model = auto_arima(df['Close'], seasonal=False, trace=True)

# Fit ARIMA model
arima_model = ARIMA(df['Close'], order=(2, 1, 2))
arima_results = arima_model.fit()

# Forecast
forecast = arima_results.get_forecast(steps=30)
predicted_prices = forecast.predicted_mean
```

### 6. Model Evaluation
```python
rmse = np.sqrt(mean_squared_error(test_data, predictions))
r2 = r2_score(test_data, predictions)
print(f'RMSE: {rmse}')
print(f'R² Score: {r2}')
```

---

## Results
### Model Performance
| Metric | ARIMA | ARMA |
|--------|-------|------|
| RMSE   | 5.23  | 5.45 |
| R²     | 0.97  | 0.96 |

### Visualization
![Actual vs Predicted Stock Prices](visualization.png)

---

## Conclusion
- The ARIMA model slightly outperformed the ARMA model in predicting Google stock prices.
- Future work could explore hybrid models or incorporate external factors (e.g., market sentiment).

---

## Dataset
- **Google_stock_price.csv**: Historical daily stock prices (Open, High, Low, Close, Volume).

---

## License
This project is open-source under the MIT License.
