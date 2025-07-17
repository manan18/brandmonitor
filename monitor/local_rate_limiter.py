import threading
import time
import os

class InMemoryRateLimiter:
    def __init__(self, rate_per_sec, burst):
        self.rate       = rate_per_sec
        self.burst      = burst
        self.tokens     = burst
        self.last_time  = time.time()
        self.lock       = threading.Lock()

    def wait_for_slot(self):
        while True:
            with self.lock:
                now = time.time()
                # Refill
                elapsed = now - self.last_time
                self.tokens = min(self.burst, self.tokens + elapsed * self.rate)
                self.last_time = now

                if self.tokens >= 1:
                    self.tokens -= 1
                    return
                # else, fall through to sleep

            time.sleep(0.05)

RATE_LIMIT_MAX = int(os.getenv("RATE_LIMIT_MAX", 800))         
RATE_INTERVAL_S = float(os.getenv("RATE_INTERVAL_S", 60))

RATE_PER_SEC = RATE_LIMIT_MAX / RATE_INTERVAL_S

shared_limiter = InMemoryRateLimiter(rate_per_sec=RATE_PER_SEC, burst=RATE_LIMIT_MAX)