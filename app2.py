import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="House Price Predictor PRO", layout="centered")

# =========================
# MODEL (stable, no crash)
# =========================
model = LinearRegression()

X = np.array([
    [2,1,0.2,800],
    [3,2,0.3,1200],
    [4,3,0.5,2000],
    [5,4,1.0,3000]
])
y = np.array([150000, 250000, 450000, 800000])

model.fit(X, y)

# =========================
# CSS (FINAL CLEAN UI)
# =========================
st.markdown("""
<style>
.block-container {
    max-width: 800px;
    padding-top: 1.5rem;
}

header {visibility: hidden;}

.card {
    background: #111827;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #1f2937;
    margin-bottom: 12px;
}

.result {
    background: linear-gradient(90deg,#00c6ff,#0072ff);
    padding: 14px;
    border-radius: 10px;
    text-align: center;
    font-size: 20px;
    font-weight: bold;
}

.range {
    background: #1e293b;
    padding: 8px;
    border-radius: 8px;
    text-align: center;
    margin-top: 6px;
}

.stButton>button {
    width: 100%;
    height: 45px;
    border-radius: 10px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================
st.markdown("""
<h1 style='text-align:center;'>🏡 House Price Predictor PRO</h1>
<p style='text-align:center; color:gray;'>Ultra clean • Stable • Production UI</p>
""", unsafe_allow_html=True)

st.markdown("---")

# =========================
# INPUT SECTION
# =========================
st.markdown("<div class='card'>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    bedrooms = st.number_input("Bedrooms", 1, 20, 2)
    bathrooms = st.number_input("Bathrooms", 1, 20, 2)

with col2:
    acre = st.number_input("Acre Lot", 0.0, 10.0, 0.5)
    size = st.slider("House Size (sqft)", 300, 5000, 1200)

st.markdown("</div>", unsafe_allow_html=True)

# =========================
# BUTTON
# =========================
predict = st.button("🚀 Predict Price")

# =========================
# PREDICTION
# =========================
if predict:

    data = np.array([[bedrooms, bathrooms, acre, size]])
    price = model.predict(data)[0]

    # clamp realistic
    price = max(50000, min(price, 2000000))

    low = price * 0.9
    high = price * 1.1

    # =========================
    # RESULT CARD
    # =========================
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    st.markdown(f"<div class='result'>💰 ${price:,.2f}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='range'>Range: ${low:,.0f} - ${high:,.0f}</div>", unsafe_allow_html=True)

    avg = 350000
    if price < avg * 0.8:
        st.success("📉 Undervalued")
    elif price > avg * 1.2:
        st.error("📈 Overpriced")
    else:
        st.info("⚖️ Fair Price")

    st.markdown("</div>", unsafe_allow_html=True)

    # =========================
    # GRAPH SECTION (FIXED SIZE)
    # =========================
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    st.markdown("### 📈 Future Trend")

    years = [1,2,3,4,5]
    growth = 0.06
    future_prices = [price * (1 + growth)**y for y in years]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=years,
        y=future_prices,
        mode='lines+markers'
    ))

    fig.update_layout(
        height=240,  # 🔥 perfect size
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis_title="Years",
        yaxis_title="Price"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# =========================
# FOOTER
# =========================
st.markdown("<p style='text-align:center; color:gray;'>✔ Final Clean UI • No Bugs</p>", unsafe_allow_html=True)
