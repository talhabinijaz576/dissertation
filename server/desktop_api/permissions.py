from django.conf import settings
from rest_framework.permissions import BasePermission

class AuthenticateDevice(BasePermission):

	def has_permssion(self, request, view):
		device_id = request.META.get('device_id')
		print("Device ID: ",device_id)
		return True