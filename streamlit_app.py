# streamlit_app.py

import os, sys

# ─── 1) Define paths ───
BASE = os.path.dirname(__file__)
SRC  = os.path.join(BASE, "src")

# ─── 2) Prep sys.path ───
# so that "import train" and "import api.main" find your src/ modules
if SRC not in sys.path:
    sys.path.insert(0, SRC)

# ─── 3) Temporarily switch cwd to src/ ───
#    so that train.py’s pd.read_csv("../data/...") resolves to BASE/data/...
_orig_cwd = os.getcwd()
os.chdir(SRC)

# ─── 4) Now import your existing modules (without touching them) ───
from train import train_and_log            # executes train.py top-level
from api.main import scaler, rf_model, cnn, rnn

# ─── 5) Restore cwd back to project root ───
os.chdir(_orig_cwd)

# ─── 6) The normal Streamlit app ───
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = os.path.join(BASE, "data", "updated_energy_dataset.csv")

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_models():
    # these were created when api.main was imported
    return scaler, rf_model, cnn, rnn

def main():
    st.set_page_config(layout="wide", page_title="🔋 Energy Predictor")
    st.title("🔋 Sustainability-Focused Energy Predictor")

    # ── Sidebar ──
    st.sidebar.header("Controls")
    if st.sidebar.button("Retrain Model"):
        with st.spinner("Running train.py…"):
            train_and_log()       # uses the same train.py you already have
        st.success("✅ Model retrained! Click Reload below.")

    if st.sidebar.button("Reload Models"):
        load_models.clear()      # clear the cache so we re-import artifacts
        st.success("🔄 Models reloaded.")

    # ── Data & Predictions ──
    df = load_data()
    st.markdown("### Raw data preview")
    st.dataframe(df.head())

    scaler_, rf_, cnn_, rnn_ = load_models()

    X   = df.drop(columns=["Target"])
    y   = df["Target"]
    Xs  = scaler_.transform(X)
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
