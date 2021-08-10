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



class logUploadView(views.APIView):

	permissions_classes = (AuthenticateDevice, )

	def post(self, request):
		headers = request.META
		print()
		pprint(headers)


		task = headers.get("HTTP_TASK", "None")
		print("Task:", task)
		
		response = {"success": True}

		if(task=="UploadSystemLogs"):
			verified, device = self.verifyDevice(headers)
			print("Verified: ", verified)
			if(verified):
				self.uploadSystemLogs(request, device)

		if(task=="UploadTaskLogs"):
			verified, response, job = self.verifyJob(headers, response)
			print("Verified: ", verified)
			if(verified):
				self.uploadTaskLogs(request, job)

		response["success"] = verified
					
		return Response(response)

	def uploadSystemLogs(self, request, device):
		try:
			print("Uploading system logs")
			filepath = getSystemLogFilePath(device, timezone.now())
			file = open(filepath, "wb")
			file.write(request.body)
			file.close()
			print("System logs uploaded successfuly")
		except Exception as e :
			print("Device ({0}) logs upload error |  {1}".format(device.device_id, e))

	def uploadTaskLogs(self, request, job):
		try:
			print("Uploading task logs")
			filepath = os.path.join(settings.LOG_ROOT, job.log_file)
			file = open(filepath, "wb")
			file.write(request.body)
			file.close()
			print("Task logs uploaded successfuly")
		except Exception as e :
			print("Task ({0}) logs upload error |  {1}".format(job.id, e))

	def verifyDevice(self, headers):
		try:
			device_id = headers.get("HTTP_DEVICE", None)
			device = Robot.objects.get(device_id=device_id)
			verified = True
		except Exception as e:
			print("Verification Error |  {0}".format(e))
			verified = False
			device = None
		return verified, device

	def verifyJob(self, headers, response):
		verified = True
		job = None
		#process = None
		try:
			task_id = int(headers["HTTP_TASKID"])
			job = Job.objects.get(id=task_id)
			if(job.device.device_id in headers.get("HTTP_DEVICE", None)):
				pass#process = job.process
			else:
				response["message"] = "Invalid device"
				verified = False
		except Exception as e:
			response["message"] = str(e)#"Invalid job"
			verified = False

		return verified, response, job


