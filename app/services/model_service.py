import joblib
import pandas as pd
from app.core.config import settings
from app.cache.redis_cache import (
    get_cached_prediction,
    set_cached_prediction
)

model = joblib.load(settings.MODEL_PATH)


def predict_car_price(data: dict):
    cache_key = str(data)

    # ---------------- Try Redis Cache ----------------
    try:
        cached = get_cached_prediction(cache_key)

        if cached:
            print("Using cached prediction")
            return float(cached)

    except Exception as e:
        print(f"Redis unavailable: {e}")


    # ---------------- Model Prediction ----------------
    input_data = pd.DataFrame([data])

    prediction = model.predict(input_data)[0]


    # ---------------- Try Save to Redis ----------------
    try:
        set_cached_prediction(cache_key, prediction)

    except Exception as e:
        print(f"Could not cache prediction: {e}")


    return float(prediction)
