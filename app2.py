import streamlit as st
import numpy as np
import xgboost as xgb
import plotly.graph_objects as go

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(page_title="House Price PRO", layout="wide")

# ----------------------------
# CLEAN PREMIUM CSS
# ----------------------------
st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1100px; margin: auto;}
.card {
    background: #111827;
    padding: 20px;
    border-radius: 14px;
    box-shadow: 0 0 15px rgba(0,255,255,0.05);
}
.title {
    text-align: center;
    font-size: 40px;
    font-weight: bold;
}
.sub {
    text-align: center;
    color: gray;
    margin-bottom: 20px;
}
.result {
    background: linear-gradient(90deg,#00c6ff,#0072ff);
    padding: 16px;
    border-radius: 12px;
    text-align: center;
    font-size: 22px;
    font-weight: bold;
}
.range {
    background: #1e293b;
    padding: 12px;
    border-radius: 10px;
    text-align: center;
    margin-top: 10px;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# LOAD MODEL (FAST + SAFE)
# ----------------------------
@st.cache_resource
def load_model():
    model = xgb.XGBRegressor()
    model.load_model("model.json")
    return model

model = load_model()

# ----------------------------
# HEADER
# ----------------------------
st.markdown('<div class="title">🏡 House Price Predictor PRO</div>', unsafe_allow_html=True)
st.markdown('<div class="sub">Ultra clean • Fast • Production ready</div>', unsafe_allow_html=True)

# ----------------------------
# MAIN CARD
# ----------------------------
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        bed = st.number_input("Bedrooms", 1, 20, 2)
        bath = st.number_input("Bathrooms", 1, 20, 2)

    with col2:
        acre = st.number_input("Acre Lot", 0.0, 10.0, 0.5)
        size = st.slider("House Size (sqft)", 300, 5000, 1200)

    st.write("")

    # ----------------------------
    # PREDICT
    # ----------------------------
    if st.button("🚀 Predict Price", use_container_width=True):

        data = np.array([[bed, bath, acre, size]])
        price = model.predict(data)[0]

        # clamp for stability
        price = max(50000, min(price, 2000000))

        low = price * 0.9
        high = price * 1.1

        st.markdown(f'<div class="result">💰 ${price:,.2f}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="range">Range: ${low:,.0f} - ${high:,.0f}</div>', unsafe_allow_html=True)

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

        # ----------------------------
        # SMALL CLEAN GRAPH
        # ----------------------------
        st.write("")
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
            height=280,  # 🔥 FIXED SIZE (no huge graph)
            margin=dict(l=10, r=10, t=20, b=10),
            xaxis_title="Years",
            yaxis_title="Price"
        )

        st.plotly_chart(fig, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------
# FOOTER
# ----------------------------
st.markdown("<br><center style='color:gray'>✔  Final Version</center>", unsafe_allow_html=True)
