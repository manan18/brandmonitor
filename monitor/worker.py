import os
import threading
import logging
from .views import worker_thread, NUM_OUTER_WORKERS

logger = logging.getLogger(__name__)

def start_workers():
    # Spawn NUM_OUTER_WORKERS daemon threads
    for i in range(NUM_OUTER_WORKERS):
        t = threading.Thread(target=worker_thread, daemon=True)
        t.start()
        logger.info(f"Started {NUM_OUTER_WORKERS} worker threads")  

