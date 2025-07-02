from django.urls import path
from .views import run_query
from .views import generate_prompts
from .views import brand_mention_score
from .views import job_status
from .views import health_check
from .views import get_total_cost
urlpatterns = [
    path('query/', run_query),
    path('generatePrompts/', generate_prompts),
    path('getMentionScore/', brand_mention_score),
    path('getJobStatus/', job_status),
    path('healthcheck/', health_check),
    path('get_total_cost/', get_total_cost)
]
