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


class applicationUpdateView(views.APIView):

	permissions_classes = (AuthenticateDevice, )

	def post(self, request):
		body = ast.literal_eval(request.body.decode("UTF-8"))
		print()
		pprint(body)


		task = body.get("task", "None")
		print("Task:", task)
		verified, device = self.verifyDevice(body)
		print("Verified: ", verified)
		response = {"success": verified}
			
		if(verified and task=="CheckApplicationUpdate"):
			response["update_required"] = self.checkUpdate(device)

		if(verified and task=="UpdateCompleted"):
			self.ChangeUpdatedDate(device)

		if(task=="GetRunnerApplication"):
			response = returnFile(settings.RUNNER_APPLICATION_ZIP)
			return response

		if(task=="GetUpdatorApplication"):
			response = returnFile(settings.UPDATOR_APPLICATION_ZIP)
			return response

		if(task=="GetAppConfig"):
			response = returnFile(settings.APP_CONFIG_FILE)
			return response


		return Response(response)

	def checkUpdate(self, device, pause_duration=1800):
		version = ApplicationVersion.objects.first()
		up_to_date = ( device.date_updated >= version.date_updated )
		last_attempt_recently = (timezone.now() - device.date_update_attempted).total_seconds() < pause_duration
		update_required = (not up_to_date) and (not last_attempt_recently)
		device.date_update_attempted = timezone.now()
		device.save()
		return update_required 

	def ChangeUpdatedDate(self, device):
		device.date_updated = timezone.now()
		device.save()
		return 

	def verifyDevice(self, body):
		try:
			device_id = body.get("device_id", None)
			device = Robot.objects.get(device_id=device_id)
			verified = True
		except Exception as e:
			print("Verification Error |  {0}".format(e))
			verified = False
			device = None
		return verified, device

