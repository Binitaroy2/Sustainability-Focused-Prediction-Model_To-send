import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn
import mlflow.tensorflow
import joblib
import os
import tempfile
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, Dropout, LSTM

# Redirect MLflow to temp dir to avoid permission issues
mlflow_tmp = os.path.join(tempfile.gettempdir(), "mlruns")
os.makedirs(mlflow_tmp, exist_ok=True)
mlflow.set_tracking_uri(f"file://{mlflow_tmp}")

# MLflow experiment
mlflow.set_experiment("energy_consumption_prediction")

df = pd.read_csv("../data/updated_energy_dataset.csv")
target = "Energy_Consumption_MWh"
selected_features = [
    'Energy_Production_MWh',
    'Type_of_Renewable_Energy',
    'Installed_Capacity_MW',
    'Energy_Storage_Capacity_MWh',
    'Storage_Efficiency_Percentage',
    'Grid_Integration_Level'
]

# Numeric features (5) for scaling, exclude categorical
numeric_features = [
    'Energy_Production_MWh',
    'Installed_Capacity_MW',
    'Energy_Storage_Capacity_MWh',
    'Storage_Efficiency_Percentage',
    'Grid_Integration_Level'
]

X = df[selected_features]
y = df[target]

# Scale only numeric features
scaler = StandardScaler()
X_numeric = X[numeric_features]
X_scaled_numeric = scaler.fit_transform(X_numeric)
X_scaled = X.copy()
X_scaled[numeric_features] = X_scaled_numeric

os.makedirs("../models", exist_ok=True)
joblib.dump(scaler, "../models/scaler.pkl")

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 1. Random Forest
with mlflow.start_run(run_name="RandomForest"):
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    preds = rf.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    mlflow.sklearn.log_model(rf, "rf_model")
    mlflow.log_metric("mse", mse)
    joblib.dump(rf, "../models/best_rf_model.pkl")
    print("RF MSE:", mse)

# Prepare for DL models
X_train_reshaped = X_train.values.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test_reshaped = X_test.values.reshape(X_test.shape[0], X_test.shape[1], 1)

# 2. CNN Model
with mlflow.start_run(run_name="CNN"):
    cnn = Sequential([
        Conv1D(64, kernel_size=3, activation='relu', padding='same', input_shape=(X_train.shape[1], 1)),
        MaxPooling1D(pool_size=2),
        Conv1D(32, kernel_size=3, activation='relu', padding='same'),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1)
    ])
    cnn.compile(optimizer='adam', loss='mse', metrics=['mae'])
    cnn.fit(X_train_reshaped, y_train, epochs=10, batch_size=32, validation_data=(X_test_reshaped, y_test), verbose=1)
    cnn_mse = cnn.evaluate(X_test_reshaped, y_test, verbose=0)[0]
    mlflow.tensorflow.log_model(cnn, "cnn_model")
    mlflow.log_metric("mse", cnn_mse)
    cnn.save("../models/cnn_model.keras")
    print("CNN MSE:", cnn_mse)

# 3. RNN Model
with mlflow.start_run(run_name="RNN"):
    rnn = Sequential([
        LSTM(64, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], 1)),
        LSTM(32, activation='relu'),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1)
    ])
    rnn.compile(optimizer='adam', loss='mse', metrics=['mae'])
    rnn.fit(X_train_reshaped, y_train, epochs=10, batch_size=32, validation_data=(X_test_reshaped, y_test), verbose=1)
    rnn_mse = rnn.evaluate(X_test_reshaped, y_test, verbose=0)[0]
    mlflow.tensorflow.log_model(rnn, "rnn_model")
    mlflow.log_metric("mse", rnn_mse)
    rnn.save("../models/rnn_model.keras")
    print("RNN MSE:", rnn_mse)