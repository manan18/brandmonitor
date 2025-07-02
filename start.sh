#!/usr/bin/env bash
set -o errexit

# Verify spaCy model installation
if ! python -c "import spacy; spacy.load('en_core_web_sm')" &>/dev/null; then
    echo "❌ Critical Error: spaCy model failed to load!"
    exit 1
fi

python manage.py migrate
# gunicorn brandmonitor.wsgi:application --bind 0.0.0.0:$PORT --timeout 6000

gunicorn brandmonitor.wsgi:application \
  -k gevent \
  --timeout 6000 \
  --keep-alive 60 \
  --bind 0.0.0.0:$PORT
