import os
import redis


redis_client = None

REDIS_URL = os.getenv("REDIS_URL")

if REDIS_URL:
    try:
        redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        redis_client.ping()
        print("Redis connected")
    except Exception as e:
        print("Redis not connected:", e)
        redis_client = None
else:
    print("Redis disabled: REDIS_URL not found")