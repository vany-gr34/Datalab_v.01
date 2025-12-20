import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

from Models.Regression.LinearRegressor import LinearRegressor
from Models.Regression.RigdeRegressor import RidgeRegressor
from Models.Regression.LassoRegressor import LassoRegressor
from Models.Regression.PolyRegressor import PolynomialRegressor
from Models.Regression.KnnRegressor import KNNRegressor
from Models.Regression.SVr import SVRRegressor
from Models.Regression.RegressorTree import DecisionTreeRegressorLab
from  Models.Regression.RandomForestRegerssor import RandomForestRegressorLab
from  Models.Regression.XgboostRegressor import XGBoostRegressor


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Data Lab – Regression Models", layout="wide")

st.title("📊 Data Lab – Regression Models Playground")
st.markdown("Test and visualize different regression models with hyperparameter search.")

# -----------------------------
# Load data
# -----------------------------
@st.cache_data
def load_data():
    data = fetch_california_housing(as_frame=True)
    X = data.data
    y = data.target
    return X, y

X, y = load_data()

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("⚙️ Configuration")

model_name = st.sidebar.selectbox(
    "Choose a model",
    [
        "Linear Regression",
        "Ridge Regression",
        "KNN Regression",
        "SVR",
        "Random Forest"
    ]
)

test_size = st.sidebar.slider("Test size", 0.1, 0.4, 0.2)

# -----------------------------
# Train / validation split
# -----------------------------
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=test_size, random_state=42
)

# -----------------------------
# Model selection
# -----------------------------
if model_name == "Linear Regression":
    model = LinearRegressor()

elif model_name == "Ridge Regression":
    model = RidgeRegressor()

elif model_name == "KNN Regression":
    model = KNNRegressor()

elif model_name == "SVR":
    model = SVRRegressor()

elif model_name == "Random Forest":
    model = RandomForestRegressorLab()

# -----------------------------
# Train button
# -----------------------------
if st.button("🚀 Train model"):
    with st.spinner("Training model & searching best hyperparameters..."):
        y_pred, metrics = model.run(
            X_train, X_valid,
            y_train, y_valid
        )

    # -----------------------------
    # Results
    # -----------------------------
    st.success("Training completed!")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📌 Best Hyperparameters")
        st.json(model.best_params)

    with col2:
        st.subheader("📈 Metrics")
        st.json(metrics)

    # -----------------------------
    # Plot: Predictions vs True
    # -----------------------------
    st.subheader("🔍 Predictions vs True Values")

    fig, ax = plt.subplots()
    ax.scatter(y_valid, y_pred, alpha=0.5)
    ax.plot([y_valid.min(), y_valid.max()],
            [y_valid.min(), y_valid.max()],
            linestyle="--")
    ax.set_xlabel("True Values")
    ax.set_ylabel("Predictions")
    ax.set_title(model_name)

    st.pyplot(fig)
