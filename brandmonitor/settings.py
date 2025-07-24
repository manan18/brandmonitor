import dj_database_url
import os
from pathlib import Path
from decouple import config  # for loading environment variables


# --- BASE SETTINGS ---
BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = config("SECRET_KEY", default="insecure-secret-for-dev")

DEBUG = config("DEBUG", default=True, cast=bool)

ALLOWED_HOSTS = ["*"]  # Update for production

# Add Render to CSRF trusted origins
CSRF_TRUSTED_ORIGINS = ['https://*.onrender.com']

# --- INSTALLED APPS ---
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',

    # Third-party
    'rest_framework',
    'corsheaders',

    # Your app
    # 'monitor',
    'monitor.apps.MonitorConfig',
    'django_celery_results'
    
]

# --- MIDDLEWARE ---
MIDDLEWARE = [
    'corsheaders.middleware.CorsMiddleware',  # must be before CommonMiddleware
    'django.middleware.common.CommonMiddleware',

    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'brandmonitor.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'brandmonitor.wsgi.application'

# --- DATABASE (Postgres for dev) ---
DATABASES = {
    'default': dj_database_url.config(
        default=config('DATABASE_URL'),
        conn_max_age=600,
        ssl_require=True
    )
}

# --- PASSWORDS / AUTH ---
AUTH_PASSWORD_VALIDATORS = []

# --- INTERNATIONALIZATION ---
LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_TZ = True

# --- STATIC FILES ---
STATIC_URL = 'static/'

# --- DEFAULT PRIMARY KEY ---
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# --- REST FRAMEWORK CONFIG (Optional) ---
REST_FRAMEWORK = {
    'DEFAULT_RENDERER_CLASSES': [
        'rest_framework.renderers.JSONRenderer',
    ],
    'DEFAULT_PARSER_CLASSES': [
        'rest_framework.parsers.JSONParser',
    ]
}

# --- CORS ---
CORS_ALLOW_ALL_ORIGINS = True  # Allow frontend requests (e.g. React, Streamlit)


CELERY_BROKER_URL = config('REDIS_URL')
CELERY_RESULT_BACKEND = 'django-db'
CELERY_WORKER_CONCURRENCY = 1  # Critical: Only 1 worker process!
CELERY_WORKER_MAX_TASKS_PER_CHILD = 10  # Recycle workers periodically

if not CELERY_BROKER_URL:
    raise ValueError("❌ REDIS_URL environment variable is not set. Please set it to your Redis instance URL.")