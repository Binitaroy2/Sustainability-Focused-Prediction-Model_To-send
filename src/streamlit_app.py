# src/streamlit_app.py

import os
import sys
import tempfile

# 1) Compute absolute paths
THIS_FILE    = os.path.abspath(__file__)                       # .../<repo>/src/streamlit_app.py
SRC_DIR      = os.path.dirname(THIS_FILE)                     # .../<repo>/src
PROJECT_ROOT = os.path.dirname(SRC_DIR)                       # .../<repo>

# 2) Redirect MLflow artifacts to a writable temp directory
mlflow_tmp = os.path.join(tempfile.gettempdir(), "mlruns")
os.makedirs(mlflow_tmp, exist_ok=True)
os.environ["MLFLOW_TRACKING_URI"] = f"file://{mlflow_tmp}"

# 3) Temporarily cd into src/ so train.py’s own pd.read_csv("../data/...") works during imports if needed
os.chdir(SRC_DIR)

# 4) Ensure we can import modules from src/
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# 5) Import your existing, untouched code
from train import train_and_log                     # src/train.py
from api.main import scaler, rf_model, cnn, rnn     # src/api/main.py

# 6) Restore cwd back to project root for the rest of the app
os.chdir(PROJECT_ROOT)

# 7) Streamlit app starts here
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "updated_energy_dataset.csv")

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_models():
    # these were initialized when we imported api.main
    return scaler, rf_model, cnn, rnn

def main():
    st.set_page_config(layout="wide", page_title="🔋 Energy Predictor")
    st.title("🔋 Sustainability‐Focused Energy Predictor")

    # Sidebar controls
    st.sidebar.header("Controls")
    if st.sidebar.button("Retrain Model"):
        with st.spinner("Running train.py…"):
            # Temporarily cd into src/ for the function call, assuming train_and_log uses relative paths like "../data/"
            original_cwd = os.getcwd()
            os.chdir(SRC_DIR)
            try:
                train_and_log()
            finally:
                os.chdir(original_cwd)
        st.success("✅ Model retrained! Click ‘Reload Models’ to pick up changes.")

    if st.sidebar.button("Reload Models"):
        load_models.clear()
        st.success("🔄 Models reloaded.")

    # Data preview
    df = load_data()
    st.markdown("### Raw data preview")
    st.dataframe(df.head())

    # Predictions
    scaler_, rf_, cnn_, rnn_ = load_models()
    X = df.drop(columns=["Target"])
    y = df["Target"]
    Xs = scaler_.transform(X)
    preds = rf_.predict(Xs)

    st.markdown("### RF: Actual vs Predicted")
    fig, ax = plt.subplots()
    ax.scatter(y, preds, alpha=0.5)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Actual vs Predicted (RF)")
    st.pyplot(fig)

if __name__ == "__main__":
    main()