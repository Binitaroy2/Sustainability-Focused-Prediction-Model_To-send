# src/streamlit_app.py

import os, sys

# ─── 1) Compute absolute paths ───
THIS_FILE   = os.path.abspath(__file__)                    # .../<repo>/src/streamlit_app.py
SRC_DIR     = os.path.dirname(THIS_FILE)                   # .../<repo>/src
PROJECT_ROOT = os.path.abspath(os.path.join(SRC_DIR, os.pardir))  # .../<repo>

# ─── 2) Switch into src/ so train.py's top-level read_csv("../data/...") works ───
os.chdir(SRC_DIR)

# ─── 3) Make sure Python can import your src modules ───
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# ─── 4) Import your untouched code ───
from train import train_and_log            # src/train.py
from api.main import scaler, rf_model, cnn, rnn  # src/api/main.py

# ─── 5) Go back to project root so all further paths are relative to it ───
os.chdir(PROJECT_ROOT)

# ─── 6) Now your normal Streamlit app ───
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "updated_energy_dataset.csv")

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_models():
    return scaler, rf_model, cnn, rnn

def main():
    st.set_page_config(layout="wide", page_title="🔋 Energy Predictor")
    st.title("🔋 Sustainability-Focused Energy Predictor")

    # Sidebar controls
    st.sidebar.header("Controls")
    if st.sidebar.button("Retrain Model"):
        with st.spinner("Running train.py…"):
            train_and_log()
        st.success("✅ Model retrained! Click Reload Models to pick up changes.")

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

    st.markdown("### RF: Actual vs. Predicted")
    fig, ax = plt.subplots()
    ax.scatter(y, preds, alpha=0.5)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Actual vs Predicted (RF)")
    st.pyplot(fig)

if __name__ == "__main__":
    main()
