
import streamlit as st
import requests

st.markdown("""
<style>
.stApp {background: linear-gradient(135deg,#000000,#1c1c1c,#2c2c2c);}
h1,h2,h3,label,.stMarkdown {color:white!important;}
div.stButton > button {
background: linear-gradient(to right,#FF512F,#DD2476);
color:white;border-radius:8px;padding:10px 20px;font-size:18px;}
.result-box {
background:#222;border:2px solid #FF512F;border-radius:10px;
padding:20px;text-align:center;color:white;font-size:24px;font-weight:bold;}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center;'>Car Price Prediction</h1>", unsafe_allow_html=True)

API_URL = "https://fastapi-project-731c.onrender.com/predict"

with st.form("prediction_form"):
    company = st.selectbox("Company",
        ["Maruti","Hyundai","Honda","Toyota","Tata","Mahindra"])
    owner = st.selectbox("Owner",["first","second","third"])
    fuel = st.selectbox("Fuel",["petrol","diesel","CNG","electric"])
    seller_type = st.selectbox("Seller Type",["Individuals","Dealers"])
    transmission = st.selectbox("Transmission",["manual","automatic"])

    km_driven = st.number_input("Kilometers Driven",0.0)
    mileage_mpg = st.number_input("Mileage",0.0)
    engine_cc = st.number_input("Engine CC",500.0)
    max_power_bhp = st.number_input("Max Power",20.0)
    torque_nm = st.number_input("Torque",50.0)
    seats = st.selectbox("Seats",[2,4,5,6,7,8,9,10])

    submit_btn = st.form_submit_button("Predict Price")

if submit_btn:
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

    try:
        response = requests.post(API_URL, json=data)
        result = response.json()

        if "predicted_price" in result:
            st.markdown(
                f"<div class='result-box'>Estimated Price:<br>{result['predicted_price']} INR</div>",
                unsafe_allow_html=True
            )
        else:
            st.error(result)

    except Exception as e:
        st.error(f"Request Failed: {e}")
