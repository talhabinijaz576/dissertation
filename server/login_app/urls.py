from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [

    path('password_reset/request', views.PasswordResetRequestView, name='password_reset_view'),
    path('password_reset/form', views.PasswordResetView, name='password_reset_view_done'),
    path('username_reset/request', views.ForgotUsernameView, name='password_reset_view'),

]
