
from django.urls import path
from . import views

urlpatterns = [
    path('train/', views.train, name='train'),
    path('generate/', views.generate_text, name='generate'),
]
