from channels.consumer import AsyncConsumer
from channels.db import database_sync_to_async
from channels.generic.websocket import WebsocketConsumer, AsyncWebsocketConsumer
import urllib.parse as urlparse
from urllib.parse import parse_qs
from portal.models import Camera, Room
from asgiref.sync import async_to_sync
import asyncio
import json
import time
from pprint import pprint


def getMessage(message):
		if(type(message)==dict): 
			message = json.dumps(message)
		return message


class PortalConsumer(WebsocketConsumer):

	def connect(self):
		#print("Websocket connected. ")
		user = self.scope["user"]
		if(user.is_authenticated):
			self.group_name = "portal_"+user.username
			async_to_sync(self.channel_layer.group_add)(self.group_name,
			 											self.channel_name)
			print("Portal Group Name: ", self.group_name)
			self.accept()
			message = {"type": "message",
					   "event": "message",
					   "message": "Connection confirmed for webpage {}".format(user.username)}
			self.send(getMessage(message))



	def disconnect(self, close_code):
		async_to_sync(self.channel_layer.group_discard)(self.group_name,
		 												self.channel_name)
		#print("Websocket disconnected.")

	def receive(self, text_data):
		print("Recieved: {}".format(text_data))


	def message(self, event):
		message = event
		self.send(getMessage(message))



class DesktopConsumer(WebsocketConsumer):

	def connect(self):

		args = parse_qs(urlparse.urlparse(self.scope['query_string'].decode('utf8')).path)
		device_number = args['device_number'][0]

		camera = Camera.objects.filter(token = device_number)
		if(camera.exists()):

			self.camera = camera[0]
			self.group_name = f"device_{device_number}"
			
			print("Accepting connection with name: ", self.group_name)
			async_to_sync(self.channel_layer.group_add)(self.group_name,
														self.channel_name)

			self.accept()
			self.camera.on_connect()

		else:

			print("Invalid device. Disconnecting...")
			self.disconnect()



	def disconnect(self, close_code):
		print("disconneting websocket", close_code)
		async_to_sync(self.channel_layer.group_discard)(self.group_name,
		 												self.channel_name)
		self.camera.on_disconnect()
		#print("Websocket disconnected.")

	def receive(self, text_data):
		print("Received: {}".format(text_data))


	def message(self, event):
		message = event
		self.send(getMessage(message))


















