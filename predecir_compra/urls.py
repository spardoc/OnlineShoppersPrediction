from . import views
from django.urls import path, re_path  # Importar re_path
from rest_framework.authtoken import views as rest_framework_views

urlpatterns = [
    
    
    # Agregar la ruta para predecir_compra
    path('predecir_compra/', views.predecir_compra, name='predecir_compra'),
]
