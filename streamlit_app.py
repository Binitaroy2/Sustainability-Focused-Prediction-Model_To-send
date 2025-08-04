# src/streamlit_app.py

import os, sys

# ─── 1) Locate directories ───
HERE         = os.path.abspath(os.path.dirname(__file__))        # .../<repo>/src
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, os.pardir))    # .../<repo>
SRC_DIR      = HERE                                              # same as above

# ─── 2) Make sure Python can import from src/ ───
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# ─── 3) Temporarily switch cwd into src/ ───
#     so train.py's top‐level `pd.read_csv("../data/...")` finds `<repo>/data/...`
_orig_cwd = os.getcwd()
os.chdir(SRC_DIR)

# ─── 4) Import your untouched modules ───
from train import train_and_log               # src/train.py
from api.main import scaler, rf_model, cnn, rnn  # src/api/main.py

# ─── 5) Restore cwd back to wherever Streamlit started (project root) ───
os.chdir(_orig_cwd)

# ─── 6) Normal Streamlit app below ───
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Absolute path to your data CSV
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "updated_energy_dataset.csv")

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_models():
    # these model objects were created when we imported api.main
    return scaler, rf_model, cnn, rnn

def main():
    st.set_page_config(layout="wide", page_title="🔋 Energy Predictor")
    st.title("🔋 Sustainability-Focused Energy Predictor")

    # ── Sidebar controls ──
    st.sidebar.header("Controls")
    if st.sidebar.button("Retrain Model"):
        with st.spinner("Running train.py…"):
            train_and_log()
        st.success("✅ Model retrained! Click Reload below.")

    if st.sidebar.button("Reload Models"):
        load_models.clear()  # drop cache so next load re-imports artifacts
        st.success("🔄 Models reloaded.")

    # ── Data preview & prediction ──
    df = load_data()
    st.markdown("### Raw data preview")
    st.dataframe(df.head())

    scaler_, rf_, cnn_, rnn_ = load_models()

    X     = df.drop(columns=["Target"])
    y_true = df["Target"]
    Xs    = scaler_.transform(X)
    y_pred = rf_.predict(Xs)

    st.markdown("### RF: Actual vs Predicted")
    fig, ax = plt.subplots()
    ax.scatter(y_true, y_pred, alpha=0.5)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Actual vs Predicted (RF)")
    st.pyplot(fig)

if __name__ == "__main__":
    main()
