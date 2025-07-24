from django.urls import path
from .views import generate_prompts
from .views import brand_mention_score
from .views import job_status
from .views import health_check
from .views import worker_check
urlpatterns = [
    path('generatePrompts/', generate_prompts),
    path('getMentionScore/', brand_mention_score),
    path('getJobStatus/', job_status),
    path('healthcheck/', health_check),
    path('worker_check/', worker_check),
]
