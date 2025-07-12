import time
from django.db import transaction
from django.utils import timezone
from .models import RateLimit

class DBRateLimiter:
    def __init__(self, key, rate_per_sec, burst):
        self.key   = key
        self.rate  = rate_per_sec
        self.burst = burst

    def wait_for_slot(self):
        while not self._try_consume():
            time.sleep(0.05)

    @transaction.atomic
    def _try_consume(self):
        now = timezone.now()
        rl, _ = RateLimit.objects.select_for_update().get_or_create(
            key=self.key,
            defaults={"tokens": self.burst, "updated_at": now}
        )
        elapsed    = (now - rl.updated_at).total_seconds()
        new_tokens = min(self.burst, rl.tokens + elapsed * self.rate)
        if new_tokens < 1.0:
            return False
        rl.tokens     = new_tokens - 1.0
        rl.updated_at = now
        rl.save(update_fields=["tokens", "updated_at"])
        return True
