from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import joblib
import numpy as np
import os
from tensorflow.keras.models import load_model

app = FastAPI()
ui_path = os.path.join(os.path.dirname(__file__), "ui")
app.mount("/ui", StaticFiles(directory=ui_path), name="ui")

FEATURES = [
    'Energy_Production_MWh',
    'Type_of_Renewable_Energy',
    'Installed_Capacity_MW',
    'Energy_Storage_Capacity_MWh',
    'Storage_Efficiency_Percentage',
    'Grid_Integration_Level'
]

numeric_features = [
    'Energy_Production_MWh',
    'Installed_Capacity_MW',
    'Energy_Storage_Capacity_MWh',
    'Storage_Efficiency_Percentage',
    'Grid_Integration_Level'
]

# Use absolute path for models
models_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models")
scaler = joblib.load(os.path.join(models_path, "scaler.pkl"))
rf_model = joblib.load(os.path.join(models_path, "best_rf_model.pkl"))
cnn_model = load_model(os.path.join(models_path, "cnn_model.keras"))
rnn_model = load_model(os.path.join(models_path, "rnn_model.keras"))

@app.get("/", response_class=HTMLResponse)
async def root():
    with open(os.path.join(ui_path, "index.html"), "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

@app.post("/predict/")
async def predict(features: dict, model_type: str = Query("rf")):
    # Convert incoming to DataFrame for scaling
    X_input = np.array([[features.get(feat, 0) for feat in FEATURES]])
    import pandas as pd
    X_df = pd.DataFrame(X_input, columns=FEATURES)
    numeric_X = X_df[numeric_features]
    scaled_numeric = scaler.transform(numeric_X)
    X_scaled = X_df.copy()
    X_scaled[numeric_features] = scaled_numeric
    X_scaled_values = X_scaled.values

    if model_type == "cnn":
        X_dl = X_scaled_values.reshape(X_scaled_values.shape[0], X_scaled_values.shape[1], 1)
        # Pad to avoid negative dimension (make timesteps at least 7)
        if X_dl.shape[1] < 7:
            pad_width = ((0,0), (0,7 - X_dl.shape[1]), (0,0))
            X_dl = np.pad(X_dl, pad_width, mode='constant')
        y_pred = cnn_model.predict(X_dl)
    elif model_type == "rnn":
        X_dl = X_scaled_values.reshape(X_scaled_values.shape[0], X_scaled_values.shape[1], 1)
        y_pred = rnn_model.predict(X_dl)
    else:
        y_pred = rf_model.predict(X_scaled_values)
    return {"prediction": float(y_pred[0][0] if len(y_pred[0]) > 0 else y_pred[0])}