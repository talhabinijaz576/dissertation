from channels.routing import ProtocolTypeRouter, URLRouter
from django.conf.urls import url
from django.urls import path, re_path
from channels.auth import AuthMiddlewareStack
from channels.security.websocket import AllowedHostsOriginValidator, OriginValidator
from portal.consumers import PortalConsumer, DesktopConsumer
from django.conf import settings


if(settings.IS_LINUX):

    application = ProtocolTypeRouter({
        'websocket': #AllowedHostsOriginValidator(
            AuthMiddlewareStack(
                URLRouter(
                [
                    url("ws/portal", PortalConsumer),
                    url("ws/app", DesktopConsumer),
                ])
            )
        ##),
    })

else:
    
    application = ProtocolTypeRouter({
        'websocket': #AllowedHostsOriginValidator(
            AuthMiddlewareStack(
                URLRouter(
                [
                    url("ws/portal", PortalConsumer.as_asgi()),
                    url("ws/app", DesktopConsumer.as_asgi()),
                ])
            )
        ##),
    })