# monitor/apps.py
import os
from django.apps import AppConfig

class MonitorConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'monitor'

    def ready(self):
        # Only run in main Django process (not in Celery workers)
        # IS_CELERY_WORKER_ON = os.getenv('IS_CELERY_WORKER_ON', 'false').lower() == 'true'
            
        # print("👷🏻  IS_CELERY_WORKER_ON:", IS_CELERY_WORKER_ON)
            
        # if not IS_CELERY_WORKER_ON :
        
        print(f"✅ Django server started (PID: {os.getpid()})")