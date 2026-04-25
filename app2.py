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
# HEADER (COMPACT)
# ----------------------------
st.markdown("""
<h1 style='text-align:center; margin-bottom:0;'>🏡 House Price Predictor PRO</h1>
<p style='text-align:center; color:gray; margin-top:0;'>Clean • Stable • Final</p>
""", unsafe_allow_html=True)

st.markdown("---")

# ----------------------------
# INPUT SECTION (TIGHT GRID)
# ----------------------------
col1, col2 = st.columns(2, gap="small")

with col1:
    bed = st.number_input("Bedrooms", 1, 20, 2)
    bath = st.number_input("Bathrooms", 1, 20, 2)

with col2:
    acre = st.number_input("Acre Lot", 0.0, 10.0, 0.5)
    size = st.slider("House Size (sqft)", 300, 5000, 1200)

# ----------------------------
# BUTTON (CENTERED + CLEAN)
# ----------------------------
st.markdown("<br>", unsafe_allow_html=True)

col_btn = st.columns([1,2,1])[1]
with col_btn:
    predict = st.button("🚀 Predict Price", use_container_width=True)

# ----------------------------
# RESULT (COMPACT + PREMIUM)
# ----------------------------
if predict:

    data = np.array([[bed, bath, acre, size]])
    price = model.predict(data)[0]

    price = max(50000, min(price, 2000000))
    low = price * 0.9
    high = price * 1.1

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(f"""
    <div style="
        background: linear-gradient(90deg,#00c6ff,#0072ff);
        padding:12px;
        border-radius:10px;
        text-align:center;
        font-size:20px;
        font-weight:bold;">
        💰 ${price:,.2f}
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="
        background:#1e293b;
        padding:8px;
        border-radius:8px;
        text-align:center;
        margin-top:6px;
        font-size:14px;">
        Range: ${low:,.0f} - ${high:,.0f}
    </div>
    """, unsafe_allow_html=True)

    # STATUS
    avg = 350000
    if price < avg * 0.8:
        st.success("📉 Undervalued")
    elif price > avg * 1.2:
        st.error("📈 Overpriced")
    else:
        st.info("⚖️ Fair Price")

    # ----------------------------
    # GRAPH (PERFECT SIZE)
    # ----------------------------
    st.markdown("---")
    st.subheader("📈 Future Trend")

    years = [1,2,3,4,5]
    growth = 0.06
    future = [price * (1 + growth)**y for y in years]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=years,
        y=future,
        mode='lines+markers'
    ))

    fig.update_layout(
        height=240,  # 🔥 tighter
        margin=dict(l=0, r=0, t=20, b=0)
    )

    st.plotly_chart(fig, use_container_width=True)
