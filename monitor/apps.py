# monitor/apps.py
from django.apps import AppConfig
import threading

class MonitorConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'monitor'

    def ready(self):
        # Delay import until apps are loaded
        from .worker import start_workers

        # Start background threads in a daemon so they don’t block shutdown
        threading.Thread(target=start_workers, daemon=True).start()


# from django.apps import AppConfig


# class MonitorConfig(AppConfig):
#     default_auto_field = 'django.db.models.BigAutoField'
#     name = 'monitor'
