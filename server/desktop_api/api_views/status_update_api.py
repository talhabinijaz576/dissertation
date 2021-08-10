from rest_framework.response import Response
from django.http import HttpResponse, HttpResponseNotFound
from rest_framework import views
from desktop_api.permissions import AuthenticateDevice
from asgiref.sync import async_to_sync, sync_to_async 
from channels.layers import get_channel_layer
from dialer.models import Robot, Job, Process, ApplicationVersion, ProcessVersion
from django.core.files import File
from django.conf import settings
from django.utils import timezone
from desktop_api.utils import *
import json
import os
import ast
import json
import mimetypes
from pprint import pprint
import datetime
from django.core.files import File
import pytz
import time



class statusUpdateView(views.APIView):

	permissions_classes = (AuthenticateDevice, )

	def post(self, request):


		body = ast.literal_eval(request.body.decode("UTF-8"))
		print()
		pprint(body)
		response = {"success": True}
		verified, response, job =  self.verify(body, response)
		print("Verified: ", verified)
		local_timezone = settings.TIMEZONES[job.device.account_owner.settings.timezone]
		if(verified):
			previous_status = job.status.lower()
			already_finished = previous_status in settings.ENDING_STATUSES
			layer = get_channel_layer()
			robot_group_name = "robot_"+job.account_owner.username

			if(not already_finished):
				
				if(job.status.lower()=="sent"):
					time.sleep(2)

				job.status = body["status"]
				if(body["status"].lower() in settings.ENDING_STATUSES):
			
					job.date_ended = timezone.now()
					if(job.date_started==None):
						job.date_started = timezone.now()
					job.device.is_running = False
					message = {"type": "message",
							   "event": "availability_update",
							   "message": "Available",
							   "background_color": "green",
							   "is_connected" : str(job.device.is_connected),
							   "robot_name": job.device.name}
					async_to_sync(layer.group_send)(robot_group_name, message)

					if(body["status"].lower()=="completed"):
						socket_message = "Job '{}' completed".format(job.name)
						notification_type = "success"
					else:
						socket_message = "Job '{}' failed".format(job.name)
						notification_type = "danger"
					message = { "type": "message",
					    		"event": "notification",
                        		"notification_type": notification_type,
								"message": socket_message}
					webuser_group_name = "webuser_"+job.account_owner.username
					async_to_sync(layer.group_send)(webuser_group_name, message)
					job.device.save()

				elif(job.status!="sent"):
					job.device.is_running = True
					job.device.save()

				if(job.status.lower()=="starting"):
					job.date_started = timezone.now()
					message = {"type": "message",
							   "event": "availability_update",
							   "message": "Busy",
							   "background_color": "red",
							   "is_connected" : str(job.device.is_connected),
							   "robot_name": job.device.name}
					async_to_sync(layer.group_send)(robot_group_name, message)

					message = { "type": "message",
					    		"event": "notification",
                        		"notification_type": "info",
								"message": "Starting job '{}'".format(job.name)}
					webuser_group_name = "webuser_"+job.account_owner.username
					async_to_sync(layer.group_send)(webuser_group_name, message)



				job.save()
				job.UpdateStatus()


			response["success"] = verified

		return Response(response)

	def verify(self, body, response):
		verified = True
		job = None
		try:
			task_id = int(body["task_id"])
			job = Job.objects.get(id=task_id)
			if(job.device.device_id not in body.get("device_id", None)):
				response["message"] = "Invalid device"
				verified = False
			elif(len(body.get("status", "")) < 1):
					response["message"] = "Invalid status message"
					verified = False
		except Exception as e:
			response["message"] = str(e)#"Invalid job"
			verified = False

		return verified, response, job


