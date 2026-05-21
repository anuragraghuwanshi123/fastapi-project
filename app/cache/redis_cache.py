import redis
from app.core.config import settings

redis_client = None

# Try connecting to Redis
if settings.REDIS_URL:
    try:
        redis_client = redis.from_url(
            settings.REDIS_URL,
            decode_responses=True
        )
        redis_client.ping()
        print("Redis connected")
    except Exception as e:
        print("Redis unavailable:", e)
        redis_client = None


def get_cached_prediction(cache_key):
    if redis_client:
        try:
            return redis_client.get(cache_key)
        except Exception as e:
            print("Cache read failed:", e)
    return None


def set_cached_prediction(cache_key, prediction):
    if redis_client:
        try:
            redis_client.setex(
                cache_key,
                3600,     # 1 hour cache
                str(prediction)
            )
        except Exception as e:
            print("Cache write failed:", e)