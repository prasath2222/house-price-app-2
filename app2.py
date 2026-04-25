import streamlit as st
import numpy as np
import xgboost as xgb
import plotly.graph_objects as go

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(page_title="House Price PRO", layout="centered")

# ----------------------------
# CSS (FIXED + STABLE)
# ----------------------------
st.markdown("""
<style>
.block-container {
    max-width: 900px;
    margin: auto;
    padding-top: 2rem;
}

h1, h2, h3 {
    text-align: center;
}

.result-box {
    background: linear-gradient(90deg,#00c6ff,#0072ff);
    padding: 15px;
    border-radius: 10px;
    text-align: center;
    font-size: 22px;
    font-weight: bold;
}

.range-box {
    background: #1e293b;
    padding: 10px;
    border-radius: 8px;
    text-align: center;
    margin-top: 8px;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# LOAD MODEL
# ----------------------------
@st.cache_resource
def load_model():
    model = xgb.XGBRegressor()
    model.load_model("model.json")
    return model

model = load_model()

# ----------------------------
# HEADER (ONLY ONCE → FIX DUPLICATE BUG)
# ----------------------------
st.title("🏡 House Price Predictor PRO")
st.caption("Ultra clean • Stable • Final version")

st.divider()

# ----------------------------
# INPUT GRID
# ----------------------------
col1, col2 = st.columns(2)

with col1:
    bed = st.number_input("Bedrooms", 1, 20, 2)
    bath = st.number_input("Bathrooms", 1, 20, 2)

with col2:
    acre = st.number_input("Acre Lot", 0.0, 10.0, 0.5)
    size = st.slider("House Size (sqft)", 300, 5000, 1200)

st.write("")

# ----------------------------
# BUTTON
# ----------------------------
if st.button("🚀 Predict Price", use_container_width=True):

    data = np.array([[bed, bath, acre, size]])
    price = model.predict(data)[0]

    # stability
    price = max(50000, min(price, 2000000))

    low = price * 0.9
    high = price * 1.1

    # ----------------------------
    # RESULT
    # ----------------------------
    st.markdown(f'<div class="result-box">💰 ${price:,.2f}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="range-box">Range: ${low:,.0f} - ${high:,.0f}</div>', unsafe_allow_html=True)

    # ----------------------------
    # STATUS
    # ----------------------------
    avg = 350000
    if price < avg * 0.8:
        st.success("📉 Undervalued")
    elif price > avg * 1.2:
        st.error("📈 Overpriced")
    else:
        st.info("⚖️ Fair Price")

    st.divider()

    # ----------------------------
    # GRAPH (FIXED SIZE)
    # ----------------------------
    st.subheader("📈 Future Trend")

    years = [1,2,3,4,5]
    growth = 0.06
    future = [price * (1 + growth)**y for y in years]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=years, y=future, mode='lines+markers'))

    fig.update_layout(
        height=260,  # 🔥 FIXED SIZE
        margin=dict(l=10, r=10, t=20, b=10)
    )

    st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# FOOTER
# ----------------------------
st.divider()
st.caption("✔ Clean stable UI • No layout bugs")
