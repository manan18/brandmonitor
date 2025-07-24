import os
from celery import Celery

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'brandmonitor.settings')

app = Celery('brandmonitor')
app.config_from_object('django.conf:settings', namespace='CELERY')

# Critical Windows-specific settings
app.conf.worker_pool = 'solo'  # Single-process pool
app.conf.worker_concurrency = 1  # Only one task at a time
app.conf.broker_transport_options = {
    'visibility_timeout': 3600,
    'max_connections': 3
}

app.autodiscover_tasks()
