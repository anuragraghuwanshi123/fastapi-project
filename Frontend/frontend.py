import streamlit as st
import requests

st.set_page_config(
    page_title="CarWise · Price Estimator",
    page_icon="🌿",
    layout="centered",
)

API_URL = "https://fastapi-project-731c.onrender.com/predict"

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@300;400;500;600;700&family=Lora:ital,wght@0,400;0,600;1,400&display=swap');

html, body, [class*="css"] {
    font-family: 'Nunito', sans-serif;
}

.stApp {
    background-color: #f5f0eb;
    background-image:
        radial-gradient(ellipse 60% 50% at 20% 0%, rgba(134,179,139,0.18) 0%, transparent 60%),
        radial-gradient(ellipse 50% 40% at 80% 100%, rgba(167,193,200,0.2) 0%, transparent 60%);
    min-height: 100vh;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding-top: 2rem;
    padding-bottom: 5rem;
    max-width: 720px;
}

/* ── HERO ── */
.hero {
    text-align: center;
    padding: 3rem 1rem 2rem;
}
.hero-leaf {
    font-size: 2.4rem;
    margin-bottom: 0.6rem;
    display: block;
    animation: float 4s ease-in-out infinite;
}
@keyframes float {
    0%, 100% { transform: translateY(0px); }
    50%       { transform: translateY(-6px); }
}
.hero-title {
    font-family: 'Lora', serif;
    font-size: clamp(2.2rem, 6vw, 3.6rem);
    font-weight: 600;
    color: #3a4a3c;
    margin: 0 0 0.4rem;
    letter-spacing: -0.01em;
}
.hero-title span {
    color: #6a9e72;
}
.hero-sub {
    font-size: 0.95rem;
    font-weight: 400;
    color: #8a9e8c;
    margin-top: 0.4rem;
    letter-spacing: 0.01em;
}

/* ── SOFT DIVIDER ── */
.soft-divider {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin: 0.5rem auto 2.2rem;
    width: 55%;
}
.soft-divider::before,
.soft-divider::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, transparent, #c5d9c7);
}
.soft-divider-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #6a9e72;
    opacity: 0.6;
}

/* ── CARD ── */
.form-card {
    background: rgba(255,255,255,0.72);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.9);
    border-radius: 20px;
    padding: 2rem 2rem 1.5rem;
    margin-bottom: 1.2rem;
    box-shadow:
        0 2px 8px rgba(100,130,105,0.07),
        0 8px 32px rgba(100,130,105,0.06);
}
.card-title {
    font-family: 'Lora', serif;
    font-size: 0.88rem;
    font-weight: 600;
    color: #6a9e72;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 1.2rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.card-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, #d4e8d6, transparent);
}

/* ── LABELS ── */
div[data-testid="stSelectbox"] label,
div[data-testid="stNumberInput"] label {
    font-family: 'Nunito', sans-serif !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    color: #7a8f7c !important;
    letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
}

/* ── SELECT ── */
div[data-testid="stSelectbox"] > div > div {
    background: #f8faf8 !important;
    border: 1.5px solid #d8e8da !important;
    border-radius: 12px !important;
    color: #3a4a3c !important;
    font-family: 'Nunito', sans-serif !important;
    font-size: 0.95rem !important;
    font-weight: 500 !important;
    box-shadow: 0 1px 4px rgba(100,130,100,0.06) !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
div[data-testid="stSelectbox"] > div > div:focus-within {
    border-color: #6a9e72 !important;
    box-shadow: 0 0 0 4px rgba(106,158,114,0.12) !important;
}

/* ── NUMBER INPUT ── */
div[data-testid="stNumberInput"] input {
    background: #f8faf8 !important;
    border: 1.5px solid #d8e8da !important;
    border-radius: 12px !important;
    color: #3a4a3c !important;
    font-family: 'Nunito', sans-serif !important;
    font-size: 0.95rem !important;
    font-weight: 500 !important;
    box-shadow: 0 1px 4px rgba(100,130,100,0.06) !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
div[data-testid="stNumberInput"] input:focus {
    border-color: #6a9e72 !important;
    box-shadow: 0 0 0 4px rgba(106,158,114,0.12) !important;
}
div[data-testid="stNumberInput"] button {
    background: #eef5ef !important;
    border-color: #d8e8da !important;
    color: #6a9e72 !important;
    border-radius: 10px !important;
}

/* ── SUBMIT BUTTON ── */
div.stButton > button {
    width: 100%;
    padding: 0.9rem 2rem;
    font-family: 'Nunito', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    color: #ffffff;
    background: linear-gradient(135deg, #7ab882 0%, #5a8e62 100%);
    border: none;
    border-radius: 14px;
    cursor: pointer;
    margin-top: 0.8rem;
    box-shadow:
        0 4px 16px rgba(90,142,98,0.3),
        0 1px 4px rgba(90,142,98,0.2);
    transition: all 0.25s ease;
}
div.stButton > button:hover {
    background: linear-gradient(135deg, #82c08a 0%, #62966a 100%);
    box-shadow:
        0 6px 24px rgba(90,142,98,0.4),
        0 2px 8px rgba(90,142,98,0.2);
    transform: translateY(-2px);
    color: #fff;
}
div.stButton > button:active {
    transform: translateY(0);
}

/* ── COLUMNS ── */
div[data-testid="column"] { padding: 0 0.35rem; }

/* ── RESULT ── */
.result-wrap {
    animation: rise 0.5s cubic-bezier(0.22,1,0.36,1);
    margin-top: 1.5rem;
}
@keyframes rise {
    from { opacity: 0; transform: translateY(20px) scale(0.98); }
    to   { opacity: 1; transform: translateY(0) scale(1); }
}
.result-card {
    background: linear-gradient(145deg, #ffffff 0%, #f0f7f1 100%);
    border: 1.5px solid #c2ddc6;
    border-radius: 20px;
    padding: 2.5rem 2rem;
    text-align: center;
    box-shadow:
        0 4px 20px rgba(90,142,98,0.1),
        0 1px 4px rgba(90,142,98,0.08),
        inset 0 1px 0 rgba(255,255,255,0.8);
    position: relative;
    overflow: hidden;
}
.result-card::before {
    content: '';
    position: absolute;
    top: 0; left: 10%; right: 10%;
    height: 3px;
    background: linear-gradient(90deg, transparent, #7ab882, #a8cfa0, #7ab882, transparent);
    border-radius: 0 0 4px 4px;
}
.result-emoji {
    font-size: 2.2rem;
    display: block;
    margin-bottom: 0.6rem;
}
.result-label {
    font-size: 0.75rem;
    font-weight: 600;
    color: #8ab48c;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
}
.result-price {
    font-family: 'Lora', serif;
    font-size: clamp(2.6rem, 8vw, 4rem);
    font-weight: 600;
    color: #3a5e3e;
    line-height: 1;
    letter-spacing: -0.02em;
}
.result-raw {
    font-size: 0.85rem;
    font-weight: 500;
    color: #9ab49c;
    margin-top: 0.6rem;
}
.result-note {
    font-size: 0.75rem;
    font-style: italic;
    color: #b8ceba;
    margin-top: 1rem;
    font-weight: 400;
}

/* ── ERROR ── */
.error-card {
    background: #fdf6f6;
    border: 1.5px solid #f0d0d0;
    border-radius: 14px;
    padding: 1rem 1.4rem;
    color: #b06060;
    font-size: 0.85rem;
    margin-top: 1rem;
    display: flex;
    align-items: center;
    gap: 0.6rem;
    box-shadow: 0 2px 8px rgba(180,100,100,0.06);
}

/* ── SPINNER ── */
div[data-testid="stSpinner"] > div {
    border-top-color: #6a9e72 !important;
}
</style>
""", unsafe_allow_html=True)

# ── HERO ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <span class="hero-leaf">🌿</span>
    <h1 class="hero-title">Car<span>Wise</span></h1>
    <p class="hero-sub">A calm, honest estimate for your car's resale value</p>
</div>
<div class="soft-divider">
    <div class="soft-divider-dot"></div>
    <div class="soft-divider-dot"></div>
    <div class="soft-divider-dot"></div>
</div>
""", unsafe_allow_html=True)

# ── FORM ─────────────────────────────────────────────────────────────────────
with st.form("prediction_form", border=False):

    # Card 1 — Vehicle
    st.markdown('<div class="form-card"><div class="card-title">🚗 &nbsp;Vehicle Details</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        company = st.selectbox("Brand", ["Maruti","Hyundai","Honda","Toyota","Tata","Mahindra"])
    with c2:
        fuel = st.selectbox("Fuel Type", ["Petrol","Diesel","CNG","Electric"])

    c3, c4 = st.columns(2)
    with c3:
        transmission = st.selectbox("Transmission", ["Manual","Automatic"])
    with c4:
        seats = st.selectbox("Seats", [2,4,5,6,7,8,9,10], index=2)
    st.markdown('</div>', unsafe_allow_html=True)

    # Card 2 — Ownership
    st.markdown('<div class="form-card"><div class="card-title">📋 &nbsp;Ownership & History</div>', unsafe_allow_html=True)
    c5, c6 = st.columns(2)
    with c5:
        owner = st.selectbox("Owner", ["First","Second","Third"])
    with c6:
        seller_type = st.selectbox("Seller Type", ["Individual","Dealer"])

    c7, _ = st.columns([1,1])
    with c7:
        km_driven = st.number_input("Kilometers Driven", min_value=0.0, step=1000.0, format="%.0f")
    st.markdown('</div>', unsafe_allow_html=True)

    # Card 3 — Specs
    st.markdown('<div class="form-card"><div class="card-title">⚙️ &nbsp;Technical Specs</div>', unsafe_allow_html=True)
    c8, c9 = st.columns(2)
    with c8:
        engine_cc = st.number_input("Engine (cc)", min_value=500.0, step=50.0, format="%.0f")
    with c9:
        max_power_bhp = st.number_input("Power (bhp)", min_value=20.0, step=5.0)

    c10, c11 = st.columns(2)
    with c10:
        torque_nm = st.number_input("Torque (Nm)", min_value=50.0, step=5.0)
    with c11:
        mileage_mpg = st.number_input("Mileage (kmpl)", min_value=0.0, step=0.5)
    st.markdown('</div>', unsafe_allow_html=True)

    submit = st.form_submit_button("🌿  Get Estimate")

# ── RESULT ───────────────────────────────────────────────────────────────────
if submit:
    payload = {
        "company":       company,
        "owner":         owner.lower(),
        "fuel":          fuel.lower(),
        "seller_type":   "Individuals" if seller_type == "Individual" else "Dealers",
        "transmission":  transmission.lower(),
        "km_driven":     km_driven,
        "mileage_mpg":   mileage_mpg,
        "engine_cc":     engine_cc,
        "max_power_bhp": max_power_bhp,
        "torque_nm":     torque_nm,
        "seats":         seats,
    }

    with st.spinner("Crunching the numbers gently…"):
        try:
            response = requests.post(API_URL, json=payload, timeout=15)
            result = response.json()

            if "predicted_price" in result:
                raw_price = str(result["predicted_price"]).replace(",", "").strip()
                p = float(raw_price)
                display = f"₹ {p/1_00_000:.2f} L" if p >= 1_00_000 else f"₹ {p:,.0f}"
                raw = f"₹ {p:,.0f} INR"

                st.markdown(f"""
                <div class="result-wrap">
                    <div class="result-card">
                        <span class="result-emoji">✨</span>
                        <p class="result-label">Estimated Resale Value</p>
                        <p class="result-price">{display}</p>
                        <p class="result-raw">{raw}</p>
                        <p class="result-note">Based on current market trends · This is an estimate, not a guarantee</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            else:
                st.markdown(f'<div class="error-card">⚠️ &nbsp; Unexpected response: {result}</div>', unsafe_allow_html=True)

        except requests.exceptions.Timeout:
            st.markdown('<div class="error-card">⏳ &nbsp; Server is waking up — please wait a moment and try again.</div>', unsafe_allow_html=True)
        except Exception as e:
            st.markdown(f'<div class="error-card">⚠️ &nbsp; Something went wrong: {e}</div>', unsafe_allow_html=True)


