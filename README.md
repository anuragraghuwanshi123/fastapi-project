

## 🚗 **AutoValuator – AI-Powered Car Price Prediction System**

Predict. Authenticate. Analyze. Deploy.

A real-time car resale price prediction system built using **XGBoost, FastAPI, Redis, and Streamlit UI** — deployed on the cloud for scalable production-grade inference.

---

### 🚀 **Live Application**

🎨 **Streamlit Frontend:**
👉 [https://car-price-ai-94xg7mpcnswbdzqftl29.streamlit.app](https://car-price-ai-94xg7mpcnswbdzqftl29.streamlit.app)

⚙️ **FastAPI Backend:**
👉 [https://fastapi-project-731c.onrender.com/docs](https://fastapi-project-731c.onrender.com/docs)

---

### 📌 **Project Overview**

AutoValuator is an end-to-end ML deployment project that brings machine learning into production with:

* 🚀 A trained **XGBoost regression model** for price prediction
* 🔐 **Protected API layer** using authentication tokens
* ⚡ **Redis caching** for faster repeated inference
* 🖥️ **Streamlit predictive UI** for end-users
* ☁️ **Cloud deployment on Render + Streamlit Cloud**

Designed for car dealerships, buyers, sellers, and valuation analysts, AutoValuator delivers fast, accurate and user-friendly pricing insights.

---

### 🧩 Features

🔑 **Token-Based Authentication**
Users must authenticate to access prediction API.

⚡ **Real-Time Prediction UI**
Frontend communicates with the FastAPI model API.

🧠 **XGBoost-Powered ML Model**
Trained on real-world vehicle dataset with tuned hyperparameters.

🚀 **Caching via Redis**
Reduces latency and accelerates repeat prediction calls.

🎛️ **Clean, Dark Modern UI**
User-friendly Streamlit interface.

---

### 🛠️ Tech Stack

| Category         | Tools / Libraries            |
| ---------------- | ---------------------------- |
| Machine Learning | XGBoost, Pandas, NumPy       |
| Backend API      | FastAPI + Uvicorn            |
| Frontend UI      | Streamlit                    |
| Caching          | Redis                        |
| Deployment       | Render, Streamlit Cloud      |
| Auth             | JWT Authentication / API Key |

---
### 🌟 Example API Request

```json
POST /predict
{
  "company": "Maruti",
  "fuel": "petrol",
  "kms_driven": 35000,
  "engine_cc": 1197,
  "power_bhp": 84,
  "year": 2018,
  "transmission": "manual",
  "owner": "first"
}
```

---
### 🧠 Future Enhancements

* 📌 CI/CD Pipeline
* 📦 Dockerization
* 📲 Flutter Mobile App
* 🧠 Model retraining automation
* 🔄 Multiple ML models with benchmarking

---

### 🤝 Contributing

Pull requests are welcome!
Feel free to open issues for improvements, bugs, or new feature proposals.

---




