from . import views
from django.urls import path, re_path  # Importar re_path
from rest_framework.authtoken import views as rest_framework_views

urlpatterns = [
    re_path(r'^get_auth_token/$', rest_framework_views.obtain_auth_token, name='get_auth_token'),
    path('predecir_compra/', views.predecir_compra, name='predecir_compra'),
     
    
]
