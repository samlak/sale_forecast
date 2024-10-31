import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb



# --- Load and Prepare the Data ---

# Load data and convert the Date column to datetime format
df = pd.read_csv('dataset.csv')
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')

# Group by 'Date' and aggregate appropriately
daily_summary = df.groupby('Date').agg({
    'Cashiers': 'sum',
    'Sales_Associates': 'sum',
    'Stockers': 'sum',
    'Customer_Service': 'sum',
    'Total_Sales': 'sum',
    'footprint': 'first',    
    'Weekly_Sales': 'first', 
    'IsHoliday': 'max',      
    'Week': 'first',         
    'Month': 'first',        
    'Type': 'first',         
    'Size': 'first'          
}).reset_index()

# Feature Engineering
daily_summary['DayOfWeek'] = daily_summary['Date'].dt.dayofweek
daily_summary['IsWeekend'] = daily_summary['DayOfWeek'].isin([5, 6]).astype(int)
daily_summary['WeekOfMonth'] = (daily_summary['Date'].dt.day - 1) // 7 + 1
daily_summary['Quarter'] = daily_summary['Date'].dt.quarter

# Lag Features
daily_summary['Total_Sales_Lag1'] = daily_summary['Total_Sales'].shift(1)
daily_summary['Total_Sales_Lag7'] = daily_summary['Total_Sales'].shift(7)
daily_summary['footprint_Lag1'] = daily_summary['footprint'].shift(1)
daily_summary['footprint_Lag7'] = daily_summary['footprint'].shift(7)

# Rolling Window Features
daily_summary['Total_Sales_Rolling7'] = daily_summary['Total_Sales'].rolling(window=7).mean()
daily_summary['footprint_Rolling7'] = daily_summary['footprint'].rolling(window=7).mean()

# Drop rows with NaN values created by lag/rolling features
data = daily_summary.dropna()



# --- Define Features and Split the Data ---

# Define features and target
features = [
    'Cashiers', 'Sales_Associates', 'Stockers', 'Customer_Service', 'footprint',
    'IsHoliday', 'Week', 'Month', 'Type', 'Size', 'DayOfWeek', 'IsWeekend',
    'WeekOfMonth', 'Quarter', 'Total_Sales_Lag1', 'Total_Sales_Lag7',
    'footprint_Lag1', 'footprint_Lag7', 'Total_Sales_Rolling7', 'footprint_Rolling7'
]
target = 'Total_Sales'

# Split into features (X) and target (y)
X = data[features]
y = data[target]

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)



# --- Train the Model ---

model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)

# Train the model
model.fit(X_train, y_train)



# --- Evaluate the Model ---

# Make predictions on the test set
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

rmse = np.sqrt(mse)
print(f"Root Mean Squared Error: {rmse}")



# --- Plot Predictions vs. Actuals ---

plt.figure(figsize=(14, 7))

# Plot the true Total Sales
plt.plot(data['Date'].iloc[-len(y_test):], y_test, label='Actual Total Sales', color='blue')

# Plot the predicted Total Sales
plt.plot(data['Date'].iloc[-len(y_test):], y_pred, label='Predicted Total Sales', color='red', linestyle='dashed')

plt.title('Actual vs Predicted Total Sales')
plt.xlabel('Date')
plt.ylabel('Total Sales')
plt.legend()
plt.show()
