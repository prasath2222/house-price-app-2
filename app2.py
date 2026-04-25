import streamlit as st
import numpy as np
import pandas as pd
import xgboost as xgb
import plotly.graph_objects as go

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(
    page_title="House Price Predictor PRO",
    page_icon="🏠",
    layout="wide"
)

# -------------------------
# LOAD MODEL
# -------------------------
@st.cache_resource
def load_model():
    model = xgb.XGBRegressor()
    model.load_model("model.json")
    return model

model = load_model()

# -------------------------
# PREMIUM CSS (GLASS UI)
# -------------------------
st.markdown("""
<style>

/* background gradient */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f172a, #020617);
}

/* center content */
.block-container {
    max-width: 900px;
    margin: auto;
}

/* glass card */
.card {
    background: rgba(255,255,255,0.05);
    padding: 20px;
    border-radius: 15px;
    backdrop-filter: blur(10px);
    margin-bottom: 20px;
}

/* button */
.stButton>button {
    width: 100%;
    border-radius: 10px;
    height: 45px;
    font-size: 16px;
    background: linear-gradient(90deg, #00ffcc, #00b3ff);
    color: black;
    border: none;
}

/* title */
.title {
    text-align: center;
    font-size: 40px;
    font-weight: bold;
}

/* subtitle */
.subtitle {
    text-align: center;
    color: #9ca3af;
    margin-bottom: 20px;
}

/* result */
.result {
    background: linear-gradient(90deg, #065f46, #064e3b);
    padding: 15px;
    border-radius: 10px;
    text-align: center;
    font-size: 20px;
}

/* sidebar */
section[data-testid="stSidebar"] {
    background-color: #020617;
}

</style>
""", unsafe_allow_html=True)

# -------------------------
# TITLE
# -------------------------
st.markdown("<div class='title'>🏠 House Price Predictor PRO</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Premium AI-powered real estate estimator</div>", unsafe_allow_html=True)

# -------------------------
# SIDEBAR
# -------------------------
st.sidebar.title("⚙️ Controls")

mode = st.sidebar.radio("Mode", ["Single Prediction", "CSV Batch"])
location = st.sidebar.selectbox("Location", ["Urban", "Suburban", "Rural"])

st.sidebar.markdown("---")
st.sidebar.write("📊 Accuracy: 0.89")
st.sidebar.write("Model: XGBoost")

# -------------------------
# LOCATION FACTOR
# -------------------------
location_factor = {
    "Urban": 1.2,
    "Suburban": 1.0,
    "Rural": 0.8
}

# -------------------------
# MAIN CARD
# -------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)

if mode == "Single Prediction":

    col1, col2 = st.columns(2)

    with col1:
        bed = st.number_input("Bedrooms", 1, 10, 2)
        bath = st.number_input("Bathrooms", 1, 10, 2)

    with col2:
        acre = st.number_input("Acre Lot", 0.01, 10.0, 0.5)
        size = st.slider("House Size (sqft)", 300, 5000, 1200)

    if st.button("Predict Price"):

        data = np.array([[bed, bath, acre, size]])
        price = model.predict(data)[0]
        price *= location_factor[location]

        st.markdown(f"<div class='result'>💰 ${price:,.2f}</div>", unsafe_allow_html=True)

        low = price * 0.9
        high = price * 1.1

        st.info(f"Estimated Range: ${low:,.0f} - ${high:,.0f}")

else:
    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:
        df = pd.read_csv(file)
        preds = model.predict(df)
        df["Predicted Price"] = preds
        st.dataframe(df)

st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# PLOTLY GRAPH (PREMIUM)
# -------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)

st.subheader("📈 Price Growth")

sizes = [500, 1000, 1500, 2000, 3000]
sample = np.array([[3, 2, 0.5, s] for s in sizes])
prices = model.predict(sample)

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=sizes,
    y=prices,
    mode='lines+markers',
    line=dict(color='#00ffcc', width=3),
    marker=dict(size=8)
))

fig.update_layout(
    template="plotly_dark",
    height=300,
    margin=dict(l=10, r=10, t=30, b=10),
    xaxis_title="Size (sqft)",
    yaxis_title="Price"
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# FOOTER
# -------------------------
st.markdown("<div style='text-align:center;color:gray;'>Built with ❤️ using XGBoost + Streamlit</div>", unsafe_allow_html=True)
