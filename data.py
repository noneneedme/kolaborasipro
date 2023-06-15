subprocess.run(['pip', 'install', 'sklearn'])
subprocess.run(['pip', 'install', 'scikit-mlm'])
import subprocess
import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
import sqlalchemy

import mysql.connector

# Membuat koneksi
df = pd.read_csv("https://raw.githubusercontent.com/noneneedme/kolaborasipro/main/SMGR.JK.csv")

# Load data
#df = pd.read_csv("/content/SMGR.JK.csv")

def predict_and_evaluate_decision_tree(df):
    # Split the data into features and target variable
    features = df[['Open', 'High', 'Low']]  # Update with the desired features
    target = df['Close']  # Update with the desired target variable
    target = target.fillna(target.mean())

    # Split the data into training and testing sets
    train_features, test_features, train_target, test_target = train_test_split(features, target, test_size=0.2, random_state=0)

    # Handle missing values
    train_features = train_features.fillna(train_features.mean())
    test_features = test_features.fillna(test_features.mean())

    # Train the model (Linear Regression)
    model = LinearRegression()
    model.fit(train_features, train_target)

    # Make predictions on the test set
    predictions = model.predict(test_features)

    # Evaluate the model
    mape = mean_absolute_percentage_error(test_target, predictions)
    r2 = r2_score(test_target, predictions)

    return mape, r2

def predict_and_evaluate_decision_tree(df):
    # Split the data into features and target variable
    features = df[['Open', 'High', 'Low']]  # Update with the desired features
    target = df['Close']  # Update with the desired target variable

    # Split the data into training and testing sets
    train_features, test_features, train_target, test_target = train_test_split(features, target, test_size=0.2, random_state=0)

    # Handle missing values
    train_features = train_features.fillna(train_features.mean())
    test_features = test_features.fillna(test_features.mean())
    train_target = train_target.fillna(train_target.mean())
    test_target = test_target.fillna(test_target.mean())

    # Train the model (Decision Tree)
    model = DecisionTreeRegressor()
    model.fit(train_features, train_target)

    # Make predictions on the test set
    predictions = model.predict(test_features)

    # Evaluate the model
    mape = mean_absolute_percentage_error(test_target, predictions)

    return mape

# Main code
st.title("Prediction and Evaluation")

# Select the model to use
model_option = st.selectbox("Select Model", ("Linear Regression", "Decision Tree"))

if model_option == "Linear Regression":
    mape, r2 = predict_and_evaluate_linear_regression(df)
    st.write("Model: Linear Regression")
    st.write("MAPE:", mape)
    st.write("R2 Score:", r2)
else:
    mape = predict_and_evaluate_decision_tree(df)
    st.write("Model: Decision Tree")
    st.write("MAPE:", mape)
