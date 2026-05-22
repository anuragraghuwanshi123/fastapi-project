import os
import json
import redis
from dotenv import load_dotenv


# ---------------- Load Environment ----------------
load_dotenv()

REDIS_URL = os.getenv("REDIS_URL")

redis_client = None


# ---------------- Redis Connection ----------------
def connect_redis():
    global redis_client

    if not REDIS_URL:
        print("⚠️ REDIS_URL not found → Running without Redis")
        return None

    try:
        redis_client = redis.StrictRedis.from_url(
            REDIS_URL,
            decode_responses=True
        )

        redis_client.ping()

        print("✅ Redis connected")

        return redis_client

    except Exception as e:

        print(f"⚠️ Redis unavailable: {e}")

        redis_client = None

        return None


# Connect at startup
connect_redis()


# ---------------- Get Cache ----------------
def get_cached_prediction(key: str):

    if redis_client is None:
        return None

    try:

        value = redis_client.get(key)

        if value:
            return json.loads(value)

    except Exception as e:

        print(f"⚠️ Cache read failed: {e}")

    return None


# ---------------- Set Cache ----------------
def set_cached_prediction(
        key: str,
        value,
        expire_time: int = 3600
):

    if redis_client is None:
        return

    try:

        redis_client.set(
            key,
            json.dumps(value),
            ex=expire_time
        )

    except Exception as e:

        print(f"⚠️ Cache write failed: {e}")


# ---------------- Delete Cache ----------------
def delete_cached_prediction(key: str):

    if redis_client is None:
        return

    try:

        redis_client.delete(key)

    except Exception as e:

        print(f"⚠️ Cache delete failed: {e}")


# ---------------- Health Check ----------------
def redis_status():

    return redis_client is not None