import streamlit as st
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="centered"   # 🔥 IMPORTANT FIX
)

# ---------------------------
# LOAD MODEL
# ---------------------------
@st.cache_resource
def load_model():
    model = xgb.XGBRegressor()
    model.load_model("model.json")
    return model

model = load_model()

# ---------------------------
# CUSTOM CSS (CLEAN UI)
# ---------------------------
st.markdown("""
<style>
.main {
    padding: 2rem;
}

.block-container {
    max-width: 800px;
    margin: auto;
}

.stButton>button {
    width: 100%;
    border-radius: 10px;
    height: 45px;
    font-size: 16px;
}

.stNumberInput, .stSlider {
    margin-bottom: 15px;
}

.result-box {
    padding: 15px;
    border-radius: 10px;
    background-color: #1e3a2f;
    text-align: center;
    font-size: 18px;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------
# TITLE
# ---------------------------
st.markdown("<h1 style='text-align:center;'>🏠 House Price Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Clean ML-based price estimator</p>", unsafe_allow_html=True)

st.markdown("---")

# ---------------------------
# INPUT SECTION
# ---------------------------
st.subheader("Enter Property Details")

col1, col2 = st.columns(2)

with col1:
    bed = st.number_input("Bedrooms", 1, 10, 2)
    bath = st.number_input("Bathrooms", 1, 10, 2)

with col2:
    acre = st.number_input("Acre Lot", 0.01, 10.0, 0.5)
    size = st.slider("House Size (sqft)", 300, 5000, 1200)

# ---------------------------
# PREDICT
# ---------------------------
if st.button("Predict Price"):

    data = np.array([[bed, bath, acre, size]])
    price = model.predict(data)[0]

    st.markdown(
        f"<div class='result-box'>💰 Predicted Price: ${price:,.2f}</div>",
        unsafe_allow_html=True
    )

    low = price * 0.9
    high = price * 1.1

    st.info(f"Estimated Range: ${low:,.0f} - ${high:,.0f}")

# ---------------------------
# GRAPH SECTION (FIXED)
# ---------------------------
st.markdown("---")
st.subheader("📊 Price vs Size")

sizes = np.array([500, 1000, 1500, 2000, 3000])
sample = np.array([[3, 2, 0.5, s] for s in sizes])
prices = model.predict(sample)

# 🔥 FIX GRAPH SIZE + ALIGNMENT
fig, ax = plt.subplots(figsize=(5, 3))  # smaller & clean

ax.plot(sizes, prices, marker='o', linewidth=2)
ax.set_title("Price Growth", fontsize=12)
ax.set_xlabel("Size (sqft)")
ax.set_ylabel("Price")

# REMOVE EXTRA WHITE SPACE
fig.tight_layout()

# CENTER GRAPH
st.pyplot(fig, use_container_width=False)

# ---------------------------
# FOOTER
# ---------------------------
st.markdown("---")
st.caption("Built with XGBoost + Streamlit")
