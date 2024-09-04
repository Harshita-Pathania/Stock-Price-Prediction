# Stock Price Prediction Model

## Introduction

This repository contains code for predicting stock prices using historical data. The goal is to forecast future stock prices based on past data to assist in investment decisions.

## Objective

The main objective is to predict future stock prices using historical data. This involves analyzing past stock prices to forecast future movements. Two machine learning models are used for this task: Linear Regression and Random Forest Regressor.

## Models Overview

**Machine Learning Models for Stock Price Prediction:**

1. **Linear Regression:**
   - **Use Case:** Suitable for modeling linear relationships between features and the target variable. It is used when there is a clear linear relationship.
   - **Advantages:** Simple to implement and interpret, computationally less intensive, and effective with smaller datasets.
   - **Limitations:** May not capture complex, non-linear patterns.

2. **Random Forest Regressor:**
   - **Use Case:** Useful for capturing non-linear relationships and interactions between features. Ideal for datasets with complex relationships.
   - **Advantages:** Handles non-linear data, provides feature importance, and is robust to overfitting.
   - **Limitations:** Computationally intensive, harder to interpret, and may overfit if not properly tuned.

## Models Selected

1. **Linear Regression:**
   - Chosen for its simplicity and effectiveness in capturing linear trends. It provides a baseline performance and works well for straightforward tasks.

2. **Random Forest Regressor:**
   - Selected to handle more complex relationships and interactions. This model captures non-linear patterns but requires careful tuning.

## How They Performed

| Model                   | Test RMSE |
|-------------------------|-----------|
| Linear Regression      | 28.37     |
| Random Forest Regressor | 273.24    |

- **Linear Regression:** Achieved a lower RMSE of 28.37, indicating better performance. Its simplicity worked well for the dataset, capturing essential trends without overfitting.

  ![image](https://github.com/user-attachments/assets/83461096-02e6-4083-9542-9b85760e493b)


- **Random Forest Regressor:** Achieved a higher RMSE of 273.24. Despite its strength in handling non-linear relationships, it performed worse in this instance, possibly due to overfitting or complexity not suited for the dataset.

  ![image](https://github.com/user-attachments/assets/36fe671a-a16d-4c0a-826b-291bcfc8ab21)


## How the Code Works

1. **Import Libraries:**
   - Libraries used include pandas, numpy, scikit-learn, and matplotlib.

2. **Load and Preprocess Data:**
   - Load data from `INFY_Historical_Data.csv`.
   - Parse dates, set 'Price Date' as the index, add moving averages (MA_5 and MA_10) as features, and split the data into training, testing, and prediction sets.

3. **Feature Engineering:**
   - Calculate 5-day and 10-day moving averages of 'Close Price'.

4. **Model Training and Testing:**
   - Train Linear Regression and Random Forest Regressor models.
   - Evaluate performance using RMSE.

5. **Prediction:**
   - Forecast future stock prices using the trained models and compare predictions with actual future prices.

6. **Visualization:**
   - Plot actual and predicted stock prices to visualize model performance.

## Usage

To run the code:

1. Install the necessary libraries:
      pip install pandas numpy scikit-learn matplotlib

2. Place the historical data in a CSV file named `INFY_Historical_Data.csv`.

3. Run the script to train the models, make predictions, and generate plots.
