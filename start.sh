#!/usr/bin/env bash
set -o errexit

# Apply database migrations
python manage.py migrate

# Start Gunicorn
gunicorn brandmonitor.wsgi:application --bind 0.0.0.0:$PORT