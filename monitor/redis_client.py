# redis_client.py
import os
import redis

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
REDIS_MAX_CONNECTIONS = int(os.getenv("REDIS_MAX_CONNECTIONS", 1000))

pool = redis.ConnectionPool.from_url(
    REDIS_URL,
    max_connections=REDIS_MAX_CONNECTIONS,
    decode_responses=False
)
client = redis.Redis(connection_pool=pool)
