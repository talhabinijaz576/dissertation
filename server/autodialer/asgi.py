"""
ASGI config for dialer project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/3.0/howto/deployment/asgi/
"""


import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'autodialer.settings')
import django
from channels.routing import get_default_application
#from django.core.asgi import get_asgi_application

django.setup()
application = get_default_application()