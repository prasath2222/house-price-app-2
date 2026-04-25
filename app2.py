import streamlit as st
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="House Price Predictor PRO",
    page_icon="🏠",
    layout="wide"
)

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():
    model = xgb.XGBRegressor()
    model.load_model("model.json")
    return model

model = load_model()

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("⚙️ Options")

mode = st.sidebar.radio(
    "Choose Mode",
    ["Single Prediction", "CSV Batch Prediction"]
)

location = st.sidebar.selectbox(
    "Select Location",
    ["Urban", "Suburban", "Rural"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Model Info")
st.sidebar.write("Accuracy (R²): 0.89")
st.sidebar.write("Model: XGBoost Regressor")

# -----------------------------
# TITLE
# -----------------------------
st.title("🏠 House Price Predictor PRO")
st.caption("Advanced ML-powered real estate estimator")

# -----------------------------
# SINGLE PREDICTION
# -----------------------------
if mode == "Single Prediction":

    col1, col2 = st.columns(2)

    with col1:
        bed = st.number_input("Bedrooms", 1, 10, 2)
        acre = st.number_input("Acre Lot", 0.01, 10.0, 0.5)

    with col2:
        bath = st.number_input("Bathrooms", 1, 10, 2)
        size = st.slider("House Size (sqft)", 300, 5000, 1200)

    # location multiplier
    location_factor = {
        "Urban": 1.2,
        "Suburban": 1.0,
        "Rural": 0.8
    }

    if st.button("Predict Price"):

        data = np.array([[bed, bath, acre, size]])

        with st.spinner("Predicting..."):
            price = model.predict(data)[0]
            price *= location_factor[location]

        st.success(f"💰 Predicted Price: ${price:,.2f}")

        # price range
        low = price * 0.9
        high = price * 1.1

        st.info(f"Estimated Range: ${low:,.0f} - ${high:,.0f}")

        # warning
        if size < 400:
            st.warning("⚠️ Very small house size → less accurate prediction")

# -----------------------------
# CSV MODE
# -----------------------------
else:
    st.subheader("📂 Upload CSV for Batch Prediction")

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:
        df = pd.read_csv(file)

        st.write("Preview:")
        st.dataframe(df.head())

        preds = model.predict(df)

        df["Predicted Price"] = preds

        st.success("Predictions added!")

        st.dataframe(df)

# -----------------------------
# GRAPH (FIXED SIZE)
# -----------------------------
st.markdown("---")
st.subheader("📈 Price vs Size")

sizes = np.array([500, 1000, 1500, 2000, 3000])
sample = np.array([[3, 2, 0.5, s] for s in sizes])
prices = model.predict(sample)

# FIX: SMALL CLEAN GRAPH
fig, ax = plt.subplots(figsize=(6, 3))  # 👈 smaller size

ax.plot(sizes, prices, marker='o')
ax.set_title("Price Growth", fontsize=12)
ax.set_xlabel("Size (sqft)")
ax.set_ylabel("Price")

st.pyplot(fig)

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.caption("Model: XGBoost Regressor | Built with Streamlit 🚀")
