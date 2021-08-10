"""
WSGI config for dialer project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/3.0/howto/deployment/wsgi/
"""

import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'autodialer.settings')
os.environ["DJANGO_ALLOW_ASYNC_UNSAFE"] = "true"

application = get_wsgi_application()
