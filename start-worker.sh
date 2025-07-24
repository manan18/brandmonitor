#!/usr/bin/env bash
set -o errexit

# Verify spaCy model (same as web service)
if ! python -c "import spacy; spacy.load('en_core_web_sm')" &>/dev/null; then
    echo "❌ Critical Error: spaCy model failed to load!"
    exit 1
fi

# Run migrations
python manage.py migrate

# Start Celery worker
celery --app=brandmonitor.celery:app worker --loglevel=info --pool=solo