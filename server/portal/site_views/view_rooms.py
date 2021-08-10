from django.shortcuts import render, redirect
from django.http import HttpResponse, HttpResponseRedirect
from django.conf import settings
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
from portal.utils.utils import *
from pprint import pprint
from django.utils import timezone
import portal.utils as utils
from django.db.models.functions import Lower
from django.contrib.auth.decorators import login_required
from django.core.files import File
from asgiref.sync import async_to_sync
from portal.models import Room, Camera
import pytz
import channels.layers
import os
import zipfile
import shutil
import time
from datetime import datetime
import random
import pandas as pd



@login_required(login_url="/accounts/login/")
def RoomsView(request, context, local_timezone):
	html_template = 'portal_rooms.html'

	if(request.POST):

		if("new_room" in request.POST):
			name = request.POST.get("room_name")
			Room.objects.create(name = name, account_owner = request.user)

		if("delete_room" in request.POST):
			id = request.POST.get("delete_room")
			rooms = Room.objects.filter(id = id, account_owner = request.user)
			if(rooms.exists()):
				rooms[0].delete()

		return HttpResponseRedirect(request.get_full_path())


	rooms = [(room, Camera.objects.filter(room = room).count()) for room in Room.objects.filter(account_owner=request.user).order_by('date_added')]
	context['records'] = rooms
	context['timezone'] = local_timezone

	context['nav_items']['Rooms'][0][0] = True
	response = render(request, template_name = html_template, context=context)
	return response



@login_required(login_url="/accounts/login/")
def CamerasView(request, context, local_timezone):
	html_template = 'portal_cameras.html'

	if(request.POST):


		if("delete_camera" in request.POST):
			id = request.POST.get("delete_camera")
			cameras = Camera.objects.filter(id = id, account_owner = request.user)
			if(cameras.exists()):
				cameras[0].delete()

		return HttpResponseRedirect(request.get_full_path())


	cameras = Camera.objects.filter(account_owner = request.user).order_by('date_added')
	cameras = [(camera, 
			    camera.date_added) for camera in cameras]
	context['records'] = cameras
	context['timezone'] = local_timezone

	context['nav_items']['Cameras'][0][0] = True
	response = render(request, template_name = html_template, context=context)
	return response






	