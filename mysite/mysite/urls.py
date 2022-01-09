"""mysite URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/dev/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import include, path
from myapp import views

urlpatterns = [
    path('polls/', include('polls.urls')),
    path('server/myapp/server/query_by_caption/<str:caption>/<str:dist_func>/<int:num_images>/', views.index),
    path('myapp/server/query_by_caption/<str:caption>/<str:dist_func>/<int:num_images>/', views.index),
    path('myapp/server/LSC_Thumbnail/<str:image_name>', views.get_image),
    path('admin/', admin.site.urls),
]
