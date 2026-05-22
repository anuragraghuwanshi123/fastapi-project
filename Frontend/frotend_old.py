import streamlit as st
import requests
import time

# ---------------------- PAGE CONFIG ----------------------
# ---------------------- CUSTOM CSS ----------------------
st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(135deg,#000000,#1c1c1c,#2c2c2c);
        }

        h1, h2, h3 {
            color: #ffffff !important;
        }

        label, .stMarkdown {
            color: #E5E5E5 !important;
        }

        /* ---------------- EXPANDER STYLING ---------------- */
        /* Background of the whole expander content */
        div.streamlit-expanderContent {
            background-color: #ffffff !important;
            padding: 15px;
            border-radius: 10px;
        }

        /* Styling text inside expander */
        div.streamlit-expanderContent label,
        div.streamlit-expanderContent p,
        div.streamlit-expanderContent h3,
        div.streamlit-expanderContent input,
        div.streamlit-expanderContent span,
        div.streamlit-expanderContent div {
            color: #000000 !important;
        }

        /* Header area of expander button */
        summary {
            background-color: #ffffff !important;
            color: #000000 !important;
            font-weight: 700;
            border-radius: 6px;
            padding: 10px;
        }

        summary:hover {
            background-color: #f2f2f2 !important;
        }

        /* Make the Generate button look consistent */
        div.stButton > button {
            background: linear-gradient(to right, #FF512F, #DD2476);
            color: white;
            border-radius: 8px;
            padding: 10px 20px;
            font-size: 18px;
            transition: 0.3s;
        }
        div.stButton > button:hover {
            transform: scale(1.05);
            background: linear-gradient(to right, #DD2476, #FF512F);
        }

        .result-box {
            background-color: #222;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            margin-top: 20px;
            border: 2px solid #FF512F;
            color: #ffffff;
            font-size: 24px;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)


# ---------------------- HEADER ----------------------
st.markdown("<h1 style='text-align:center;'> Car Price Prediction</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center; color:#d1d1d1;'>Enter required details below and get the estimated resale value.</h4><br>", unsafe_allow_html=True)

# ---------------------- API URL ----------------------
API_URL = "https://fastapi-project-731c.onrender.com/predict"
LOGIN_URL = "https://fastapi-project-731c.onrender.com/login"

# ---------------------- AUTH SECTION ----------------------
with st.expander("🔐 API Authentication (Required)", expanded=False):
    st.markdown("<p class='auth-title'>Authentication Required To Access Prediction API</p>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)

    with col1:
        api_key = st.text_input("API Key", value=st.session_state.get("apikey", ""), type="password")

    with col2:
        st.subheader("Get Token Automatically")
        username = st.text_input("Username", key="username")
        password = st.text_input("Password", key="password", type="password")

        if st.button("Generate Token"):
            if not username or not password:
                st.error("Enter username & password!")
            else:
                try:
                    resp = requests.post(LOGIN_URL, json={"username": username, "password": password})
                    if resp.status_code == 200:
                        result = resp.json()
                        if "access_token" in result:
                            token = result["access_token"]
                            st.session_state["token"] = token
                            st.session_state["token_expiry"] = time.time() + 1800  # 30 minutes
                            st.session_state["apikey"] = api_key
                            st.success(" Token generated successfully! (Valid 30 mins)")
                        else:
                            st.error(" Invalid credentials")
                    else:
                        st.error(f"Failed to generate token: {resp.text}")
                except Exception as e:
                    st.error(f"Error: {e}")

# ---------------------- USER INPUT FORM ----------------------
with st.form("prediction_form"):
    st.subheader(" Car Details")

    company = st.selectbox(
        "Car Brand (Company)", 
        ["Maruti", "Hyundai", "Honda", "Toyota", "Tata", "Mahindra", "Ford", "Kia",
         "Renault", "Volkswagen", "Skoda", "Nissan", "MG", "Jeep", "Mercedes",
         "BMW", "Audi", "Jaguar", "Volvo"]
    )
    
    col3, col4 = st.columns(2)
    with col3:
        owner = st.selectbox("Owner Type", ["first", "second", "third"])
        fuel = st.selectbox("Fuel Type", ["petrol", "diesel", "CNG", "electric"])
        seller_type = st.selectbox("Seller Type", ["Individuals", "Dealers"])
    with col4:
        transmission = st.selectbox("Transmission", ["manual", "automatic"])
        km_driven = st.number_input("Kilometers Driven", min_value=0.0)
        mileage_mpg = st.number_input("Mileage (MPG)", min_value=0.0)

    engine_cc = st.number_input("Engine CC", min_value=500.0)
    max_power_bhp = st.number_input("Max Power (BHP)", min_value=20.0)
    torque_nm = st.number_input("Torque (Nm)", min_value=50.0)

    # -------- Updated: Seats Dropdown --------
    seats = st.selectbox("Seats", [2, 4, 5, 6, 7, 8, 9, 10])

    submit_btn = st.form_submit_button(" Predict Price")

# ---------------------- TOKEN WARNING ----------------------
if "token_expiry" in st.session_state:
    remaining = int(st.session_state["token_expiry"] - time.time())
    if remaining <= 0:
        st.warning(" Token expired! Please generate a new one.")
    elif remaining < 300:
        st.warning(f" Token will expire in {remaining} seconds. Consider regenerating it.")

# ---------------------- API CALL ----------------------
if submit_btn:
    if "token" not in st.session_state:
        st.error("⚠ Please generate token before predicting.")
    else:
        data = {
            "company": company,
            "owner": owner,
            "fuel": fuel,
            "seller_type": seller_type,
            "transmission": transmission,
            "km_driven": km_driven,
            "mileage_mpg": mileage_mpg,
            "engine_cc": engine_cc,
            "max_power_bhp": max_power_bhp,
            "torque_nm": torque_nm,
            "seats": seats
        }

        headers = {
            "token": st.session_state["token"],
            "api-key": st.session_state.get("apikey", "")
        }

        try:
            response = requests.post(API_URL, json=data, headers=headers)
            result = response.json()

            if "predicted_price" in result:
                st.markdown(f"<div class='result-box'> Estimated Price: <br>{result['predicted_price']} INR</div>", unsafe_allow_html=True)
            else:
                st.error(f" Error: {result}")

        except Exception as e:
            st.error(f" Request Failed: {e}")
