import uuid
from django.db import models
from django.utils import timezone

class Job(models.Model):
    STATUS_QUEUED     = "queued"
    STATUS_PROCESSING = "processing"
    STATUS_COMPLETED  = "completed"
    STATUS_FAILED     = "failed"

    job_id       = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    brand        = models.CharField(max_length=200)
    prompts      = models.JSONField()
    status       = models.CharField(max_length=20, default=STATUS_QUEUED)
    progress     = models.FloatField(default=0.0)
    result       = models.JSONField(null=True, blank=True)
    created_at   = models.DateTimeField(auto_now_add=True)
    started_at   = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    error        = models.TextField(null=True, blank=True)

class RateLimit(models.Model):
    key        = models.CharField(max_length=100, primary_key=True)
    tokens     = models.FloatField()
    updated_at = models.DateTimeField()
