from django.contrib import admin
from django.urls import path, include
from monitor.views import home

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', home),
    path('api/', include('monitor.urls')),  # Your API endpoint
]
