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



class rundialerTaskView(views.APIView):

	permissions_classes = (AuthenticateDevice, )


	def post(self, request):
		body = ast.literal_eval(request.body.decode("UTF-8"))
		pprint(body)
		response = {"success": True}
		task = body.get("task", "None")
		#print("Task:", task)

		
		if(task=="GetPythonDownloadUrl"):
			version = body.get("version", settings.DEFAULT_PYTHON_VERSION)
			response["python_download_url"] = settings.PYTHON_DOWNLOAD_URLS.get(version, settings.DEFAULT_PYTHON_DOWNLOAD_URL_PATTERN.format(version))
			response["pip_download_url"] = settings.PIP_DOWNLOAD_URLS.get(version, settings.DEFAULT_PIP_FILE)


		if(task=="GetDefaultRequirementsTxt"):
			response = returnFile(settings.DEFAULT_REQUIREMENTS_TXT)
			return response

		if(task=="GetUipathConfig"):
			device = Robot.objects.get(device_id=body.get("device_id", "None"))
			response["uirobot_path"] = device.uirobot_path
			response["uipath_log_folder"] = device.uipath_log_folder

		verified, response, job, process, process_folder =  self.verify(body, response)
		#print("Verified: ", verified)
		if(not verified):
			response["success"] = False
			return Response(response)


		if(task=="GetProcessFileStructure"):
			response = self.GetProcessFileStructure(response, process_folder)
			response["argument_string"] = self.GetArgumentString(job)

		
		if(task=="GetTaskFile"):
			filename = body["filename"]
			directory = os.path.join(settings.CODE_ROOT, process.account_owner.username, str(process.id), process.current_version)
			if(not os.path.exists(directory)):
				directory = os.path.join(settings.CODE_ROOT, process.account_owner.username, process.process_name, process.current_version)
			file_path = os.path.join(directory, filename)
			#print("File_path: {}".format(file_path))
			response = returnFile(file_path)
			return response

		#pprint(response)
		#print()

		return Response(response)

	def verify(self, body, response):
		verified = True
		job = None
		process = None
		process_folder = None
		try:
			task_id = int(body["task_id"])
			job = Job.objects.get(id=task_id)
			local_timezone = settings.TIMEZONES[job.account_owner.settings.timezone]

			if(job.device.device_id in body.get("device_id", None)):
				process = job.process
				response["process_name"] = job.process.process_name
				response["process_type"] = job.process.process_type
				response["timeout"] = job.timeout
				process_last_modified = job.process.date_modified.astimezone(local_timezone).replace(tzinfo=None).strftime("%Y-%m-%d %H:%M:%S") if job.process.date_modified!=None else ""
				response["process_last_modified"] = process_last_modified
				try:
					response["python_version"] = job.process.python_version
					if(len(response["python_version"] )<4): raise Exception("Error")
				except:	
					response["python_version"] = settings.DEFAULT_PYTHON_VERSION
				process_folder = os.path.join(settings.CODE_ROOT, job.account_owner.username, str(job.process.id), job.process.current_version)
			else:
				response["message"] = "Invalid device"
				verified = False
		except Exception as e:
			response["python_version"] = settings.DEFAULT_PYTHON_VERSION
			response["message"] = str(e)#"Invalid job"
			verified = False

		return verified, response, job, process, process_folder

	def GetProcessFileStructure(self, response, process_folder):
		structure_file =  os.path.join(process_folder, settings.DIRECTORY_STRUCTURE_FILENAME)
		print("File_structure_file: ", structure_file)
		json_dict = json.load(open(structure_file))
		response["main_file"] = json_dict["main_file"]
		response["directories"] = json_dict["directories"]
		response["files"] = json_dict["files"]
		#response = returnFile(structure_file)
		return response


	def GetArgumentString(self, job):
		#process_version = ProcessVersion.objects.get(process_name = self.job.process, process_version = self.job.process.current_version)
		argument_string =  ""

		if(job.process.process_type.lower()=="python"):
			for arg in job.arguments.split(";"):
				try:
					key, value = arg.split(",")
					
					key = key.strip()
					if(key[0] in ["'", '"']):
						key = key[1:]
					if(key[-1] in ["'", '"']):
						key = key[:-1]
					key = key.replace('"', "'")

					value = value.strip()
					if(value[0] in ["'", '"']):
						value = value[1:]
					if(value[-1] in ["'", '"']):
						value = value[:-1]
					value = value.replace('"', "'")
					argument_string = argument_string + ' "{0}" "{1}"'.format(key, value)
				except:
					pass

		elif(job.process.process_type.lower()=="uipath"):
			argument_string = self.GenerateUiPathArgumentString(job.arguments)

		else:
			argument_string = job.arguments

		return argument_string


	def GenerateUiPathArgumentString(self, arguments):
		try:
			arguments = arguments.replace('"', "'").strip()
			argument_string = '"{'
			args_present = False
			for arg in arguments.split(";"):
				try:
					key, value = arg.split(",")
					if(value[0]!="'"):
						temp_value = value.lower().strip()
						if((temp_value not in ['true', 'false']) and not temp_value.isnumeric()):
							value = "'" + value
							if(value[-1]!="'"):
								value = value + "'"
					if(value.lower() in ['true', 'false']):
						value = value.lower()
								
					argument_string = argument_string + "'{}': {} , ".format(key, value)
					args_present = True
				except:
					pass

			if(args_present):	
				argument_string  = argument_string[:-3] + '}"'
			else:
				argument_string  = argument_string + '}"'
		except:
			argument_string = "{}"

		return argument_string




