import joblib
import pandas as pd
from app.core.config import settings
from app.cache.redis_cache import set_cached_prediction, get_cached_prediction

model = joblib.load(settings.MODEL_PATH)


def predict_car_price(data: dict):
    cache_key = str(data)

    # 1. Try cache safely
    try:
        cached = get_cached_prediction(cache_key)
        if cached is not None:
            return float(cached)
    except Exception as e:
        print("Redis cache read failed:", e)

    # 2. Model prediction
    input_data = pd.DataFrame([data])
    prediction = model.predict(input_data)[0]

    # 3. Save cache safely
    try:
        set_cached_prediction(cache_key, prediction)
    except Exception as e:
        print("Redis cache write failed:", e)

    return prediction
