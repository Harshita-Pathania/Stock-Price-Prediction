# Step 1: Import the necessary libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Step 2: Load and preprocess the data
data = pd.read_csv('INFY_Historical_Data.csv', parse_dates=['Price Date'], dayfirst=True)
data.set_index('Price Date', inplace=True)

# Sort the data by the index (Price Date) to ensure it is in ascending order
data.sort_index(inplace=True)

# Step 3: Feature Engineering
# Add moving averages as features
data['MA_5'] = data['Close Price'].rolling(window=5).mean()
data['MA_10'] = data['Close Price'].rolling(window=10).mean()
data.dropna(inplace=True)

# Step 4: Split the data into training (first 4 years), testing (next 4 years), and prediction (last 2 years) sets
# Define the split dates
train_end_date = '2017-12-31'
test_start_date = '2018-01-01'
test_end_date = '2021-12-31'
predict_start_date = '2022-01-01'
predict_end_date = '2023-12-31'  

# Split the data
train_data = data.loc[:train_end_date]
test_data = data.loc[test_start_date:test_end_date]
predict_data = data.loc[predict_start_date:predict_end_date]

# Prepare the data for training and testing
features = ['MA_5', 'MA_10']

train_X = train_data[features].values
train_y = train_data['Close Price'].values

test_X = test_data[features].values
test_y = test_data['Close Price'].values

# Step 5: Train the model
model = LinearRegression()
model.fit(train_X, train_y)

# Step 6: Test the model
test_predictions = model.predict(test_X)
test_rmse = np.sqrt(mean_squared_error(test_y, test_predictions))
print(f'Test RMSE: {test_rmse}')

# Step 7: Predict future stock prices using past data
predicted_prices = []
current_data = test_data.copy()

for date in predict_data.index:
    X_predict = current_data[features].values[-1].reshape(1, -1)
    prediction = model.predict(X_predict)[0]
    predicted_prices.append(prediction)
    
    # Add the new prediction to the current_data for future predictions
    new_row = predict_data.loc[[date]].copy()
    new_row['Close Price'] = prediction
    current_data = pd.concat([current_data, new_row])

# Add predictions to the predict_data dataframe
predict_data['Predicted Price'] = predicted_prices

# Step 8: Plot the predicted and actual stock prices
plt.figure(figsize=(14, 7))
plt.plot(train_data.index, train_y, label='Training Data')
plt.plot(test_data.index, test_y, label='Testing Data')
plt.plot(test_data.index, test_predictions, label='Test Predictions', linestyle='dashed')
plt.plot(predict_data.index, predict_data['Close Price'], label='Actual Future Data')
plt.plot(predict_data.index, predict_data['Predicted Price'], label='Future Predictions', linestyle='dashed')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.title('Stock Price Prediction')
plt.show()
