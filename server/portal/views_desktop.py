from rest_framework.response import Response
from django.http import HttpResponse, HttpResponseNotFound, JsonResponse, Http404
from django.views.decorators.csrf import csrf_exempt
from rest_framework import views
from desktop_api.permissions import AuthenticateDevice
from asgiref.sync import async_to_sync, sync_to_async 
from channels.layers import get_channel_layer
from django.shortcuts import render, redirect
from portal.site_views.view_rooms import RoomsView, CamerasView
from portal.site_views.view_employees import EmployeesView
from portal.models import Room, Camera, Employee, Picture, Permission
from django.core.files import File
from django.conf import settings
from django.utils import timezone
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




def send_reload_message(user):

	layer = get_channel_layer()
	message = {"type": "message",
				"event": "reload_signal"}

	group_name = "portal_" + user.username
	print(f"Sending reload to {group_name}")
	async_to_sync(layer.group_send)(group_name, message)



@csrf_exempt
def DesktopAppView(request, device_token):


	response = {"success" : True}

	if(request.POST):

		pprint(request.POST)

		if("register_device" in request.POST):

			code = request.POST.get("activation_code")
			rooms = Room.objects.filter(code = code)

			if(rooms.exists()):

				room = rooms[0]
				cameras = Camera.objects.filter(account_owner = room.account_owner)
				camera_names = [camera.name for camera in cameras]

				i = 1
				while(True):
					name = f"Camera {i}"
					if(name not in camera_names):
						break
					i = i+1

				Camera.objects.create(name = name, 
									  token = device_token,
									  room = room, 
									  account_owner = room.account_owner)

				send_reload_message(room.account_owner)

				return JsonResponse(response)

			return HttpResponseNotFound()



	if(request.GET):

		#pprint(request.META)

		if("get_images_list" in request.GET):

			employee_dict = {}
			camera = Camera.objects.filter(token = device_token)

			if(camera.exists()):
				camera = camera[0]
				employees = Employee.objects.filter(account_owner = camera.account_owner)
				print(employees)
				for employee in employees:
					images = [image.filepath for image in Picture.objects.filter(employee = employee)]
					images = list(map(lambda x: f"/static/pictures/{x}", images))
					employee_dict[str(employee.id)] = {"name" : employee.name, "images" : images, }
				camera.date_synced = timezone.now()
				camera.save()
			
			#print("returning employee list")
			#pprint(employee_dict)
			return JsonResponse(employee_dict)


		if("request_access" in request.GET):

			has_access = False
			employee_id = int(request.GET.get("request_access"))

			try:
				employee = Employee.objects.get(id = employee_id)
				camera = Camera.objects.filter(token = device_token)[0]
				has_access = camera.check_access(employee)
				camera.raise_event(True, has_access, employee.name)	
			except Exception as e:
				try:
					camera = Camera.objects.filter(token = device_token)[0]
					camera.raise_event(False, False, "Unknown Person")	
				except:
					pass
				print(e)

			return JsonResponse({"access_granted" : has_access})





	return JsonResponse(response)






