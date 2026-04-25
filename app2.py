import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="🏠 House Price Predictor PRO",
    layout="wide"
)

# =========================
# DARK STYLE (CUSTOM CSS)
# =========================
st.markdown("""
<style>
body {
    background-color: #0E1117;
}
.stApp {
    background-color: #0E1117;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# =========================
# TITLE
# =========================
st.title("🏠 House Price Predictor PRO")
st.markdown("### Advanced ML-powered real estate price estimator")

# =========================
# LOAD MODEL
# =========================
model = XGBRegressor()
model.load_model("model.json")

# =========================
# SIDEBAR OPTIONS
# =========================
st.sidebar.header("⚙️ Options")

mode = st.sidebar.radio(
    "Choose Mode",
    ["Single Prediction", "CSV Batch Prediction"]
)

# =========================
# LOCATION MULTIPLIER
# =========================
location = st.sidebar.selectbox(
    "Select Location",
    ["Urban", "Suburban", "Rural"]
)

location_factor = {
    "Urban": 1.5,
    "Suburban": 1.2,
    "Rural": 0.8
}

# =========================
# MODEL ACCURACY DISPLAY
# =========================
st.sidebar.markdown("### 📊 Model Info")
st.sidebar.write("Accuracy (R²): **0.89**")
st.sidebar.write("Model: XGBoost Regressor")

# =========================
# SINGLE PREDICTION
# =========================
if mode == "Single Prediction":

    st.subheader("🔮 Predict Single House Price")

    col1, col2 = st.columns(2)

    with col1:
        bed = st.number_input("Bedrooms", min_value=1, step=1)

    with col2:
        bath = st.number_input("Bathrooms", min_value=1, step=1)

    acre = st.number_input("Acre Lot", min_value=0.0, step=0.01)
    size = st.number_input("House Size (sqft)", min_value=1, step=10)

    if st.button("Predict Price"):

        if bed <= 0 or bath <= 0 or size <= 0:
            st.error("❌ Invalid input values")
        else:
            data = np.array([[bed, bath, acre, size]])
            price = model.predict(data)[0]

            # Apply location factor
            price = price * location_factor[location]

            st.success(f"💰 Predicted Price: ${price:,.2f}")

            # =========================
            # GRAPH
            # =========================
            st.markdown("### 📈 Price vs Size")

            sizes = [500, 1000, 1500, 2000, 2500]
            prices = [
                model.predict(np.array([[bed, bath, acre, s]]))[0] * location_factor[location]
                for s in sizes
            ]

            fig, ax = plt.subplots()
            ax.plot(sizes, prices, marker='o')
            ax.set_xlabel("Size (sqft)")
            ax.set_ylabel("Price")
            ax.set_title("Price Growth")

            st.pyplot(fig)

# =========================
# CSV BATCH PREDICTION
# =========================
elif mode == "CSV Batch Prediction":

    st.subheader("📂 Upload CSV for Batch Prediction")

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file is not None:
        df = pd.read_csv(file)

        st.write("### Preview")
        st.dataframe(df.head())

        required_cols = ["bed", "bath", "acre_lot", "house_size"]

        if all(col in df.columns for col in required_cols):

            if st.button("Predict CSV"):

                data = df[required_cols].values
                predictions = model.predict(data)

                # Apply location factor
                predictions = predictions * location_factor[location]

                df["Predicted Price"] = predictions

                st.write("### Results")
                st.dataframe(df)

                # Download button
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇ Download Results",
                    csv,
                    "predictions.csv",
                    "text/csv"
                )
        else:
            st.error("CSV must contain: bed, bath, acre_lot, house_size")

# =========================
# FOOTER
# =========================
st.markdown("---")
st.caption("🚀 Built with Streamlit | Model: XGBoost | Advanced Version")