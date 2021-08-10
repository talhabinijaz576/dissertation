from channels.routing import route_class
from .consumers import PortalConsumer

channel_routing = [
	route_class(PortalConsumer)
]