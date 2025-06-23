from django.urls import path
from .views import run_query
from .views import generate_prompts
from .views import brand_mention_score

urlpatterns = [
    path('query/', run_query),
    path('generatePrompts/', generate_prompts),
    path('getMentionScore/', brand_mention_score)
]
