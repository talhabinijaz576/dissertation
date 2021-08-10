from django.urls import path
from . import views
from desktop_api.api_views.run_task_api import rundialerTaskView
from desktop_api.api_views.status_update_api import statusUpdateView
from desktop_api.api_views.log_upload_api import logUploadView
from desktop_api.api_views.application_update_api import applicationUpdateView
from desktop_api.api_views.settings_api import settingsUpdateView


urlpatterns = [
    path('runTask', rundialerTaskView.as_view(), name='run_task'),
    path('logUpload', logUploadView.as_view(), name='log_upload'),
    path('statusUpdate', statusUpdateView.as_view(), name='status_update'),
    path('settingsUpdate', settingsUpdateView.as_view(), name='settings_update'),
    path('applicationUpdate', applicationUpdateView.as_view(), name='application_update'),
    
]
