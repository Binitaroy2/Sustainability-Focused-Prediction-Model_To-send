# src/streamlit_app.py

import os
import sys
import tempfile

# ─── 1) Determine absolute paths ───
THIS_FILE    = os.path.abspath(__file__)           # .../<repo>/src/streamlit_app.py
SRC_DIR      = os.path.dirname(THIS_FILE)          # .../<repo>/src
PROJECT_ROOT = os.path.dirname(SRC_DIR)            # .../<repo>

# ─── 2) Redirect MLflow to a writable temp dir ───
#    Any mlflow.sklearn.log_model() at import time in train.py
#    will now write to /tmp/mlruns (or equivalent) instead of your repo.
mlflow_temp = os.path.join(tempfile.gettempdir(), "mlruns")
os.makedirs(mlflow_temp, exist_ok=True)
os.environ["MLFLOW_TRACKING_URI"] = f"file://{mlflow_temp}"

# ─── 3) Temporarily switch cwd into src/ ───
#    So that train.py’s pd.read_csv("../data/...") finds <repo>/data/…
os.chdir(SRC_DIR)

# ─── 4) Ensure Python can import your src modules ───
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# ─── 5) Import your untouched modules ───
from train import train_and_log                    # src/train.py
from api.main import scaler, rf_model, cnn, rnn     # src/api/main.py

# ─── 6) Restore cwd back to project root ───
os.chdir(PROJECT_ROOT)


# ─── 7) Now the normal Streamlit app ───
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "updated_energy_dataset.csv")

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_models():
    # These were initialized when api.main was imported
    return scaler, rf_model, cnn, rnn

def main():
    st.set_page_config(layout="wide", page_title="🔋 Energy Predictor")
    st.title("🔋 Sustainability-Focused Energy Predictor")

    # ─── Sidebar controls ───
    st.sidebar.header("Controls")
    if st.sidebar.button("Retrain Model"):
        with st.spinner("Running train.py…"):
            train_and_log()   # will use MLflow temp dir
        st.success("✅ Model retrained! Click ‘Reload Models’ to pick up changes.")

    if st.sidebar.button("Reload Models"):
        load_models.clear()  # drop cache so we re-import updated artifacts
        st.success("🔄 Models reloaded.")

    # ─── Data preview & predictions ───
    df = load_data()
    st.markdown("### Raw data preview")
    st.dataframe(df.head())

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
