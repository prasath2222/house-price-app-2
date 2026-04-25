import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

# -----------------------
# CONFIG
# -----------------------
st.set_page_config(page_title="House Price Predictor PRO", layout="centered")

# -----------------------
# MODEL
# -----------------------
model = LinearRegression()

X = np.array([
    [2,1,0.2,800],
    [3,2,0.3,1200],
    [4,3,0.5,2000],
    [5,4,1.0,3000]
])
y = np.array([150000, 250000, 450000, 800000])

model.fit(X, y)

# -----------------------
# STYLE (NO EXTRA BOXES)
# -----------------------
st.markdown("""
<style>
.block-container {
    max-width: 700px;
    padding-top: 1rem;
}
header {visibility:hidden;}

.title {
    text-align:center;
    font-size:28px;
    font-weight:700;
}
.sub {
    text-align:center;
    color:gray;
    margin-bottom:15px;
}

.result {
    background: linear-gradient(90deg,#00c6ff,#0072ff);
    padding: 12px;
    border-radius: 8px;
    text-align: center;
    font-size: 18px;
    font-weight: bold;
    margin-top:10px;
}

.range {
    text-align:center;
    color:lightgray;
    margin-top:5px;
}

.stButton>button {
    width:100%;
    border-radius:8px;
    height:40px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------
# HEADER
# -----------------------
st.markdown("<div class='title'>🏡 House Price Predictor PRO</div>", unsafe_allow_html=True)
st.markdown("<div class='sub'>Clean • Stable • Final</div>", unsafe_allow_html=True)

st.divider()

# -----------------------
# INPUTS (NO CARDS = CLEAN)
# -----------------------
col1, col2 = st.columns(2)

with col1:
    bedrooms = st.number_input("Bedrooms", 1, 20, 2)
    bathrooms = st.number_input("Bathrooms", 1, 20, 2)

with col2:
    acre = st.number_input("Acre Lot", 0.0, 10.0, 0.5)
    size = st.slider("House Size (sqft)", 300, 5000, 1200)

# -----------------------
# BUTTON
# -----------------------
predict = st.button("🚀 Predict Price")

# -----------------------
# RESULT
# -----------------------
if predict:

    data = np.array([[bedrooms, bathrooms, acre, size]])
    price = model.predict(data)[0]

    price = max(50000, min(price, 2000000))

    low = price * 0.9
    high = price * 1.1

    st.markdown(f"<div class='result'>💰 ${price:,.2f}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='range'>Range: ${low:,.0f} - ${high:,.0f}</div>", unsafe_allow_html=True)

    avg = 350000
    if price < avg * 0.8:
        st.success("Undervalued")
    elif price > avg * 1.2:
        st.error("Overpriced")
    else:
        st.info("Fair Price")

    # -----------------------
    # GRAPH (PERFECT SIZE)
    # -----------------------
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
        height=220,   # 🔥 small clean graph
        margin=dict(l=0, r=0, t=10, b=0)
    )

    st.plotly_chart(fig, use_container_width=True)
