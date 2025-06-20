from django.urls import path
from .views import run_query
from .views import generate_prompts

urlpatterns = [
    path('query/', run_query),
    path('generatePrompts/', generate_prompts)
]
