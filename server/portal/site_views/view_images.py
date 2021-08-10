from django.shortcuts import render, redirect
from django.http import HttpResponse, HttpResponseRedirect
from django.conf import settings
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
from portal.utils.utils import *
from pprint import pprint
from django.utils import timezone
import portal.utils as utils
from django.db.models.functions import Lower
from django.core.files import File
from asgiref.sync import async_to_sync
from portal.models import Room, Camera, Employee, Permission, Picture
from portal.views import getDefaultContext
import pytz
import channels.layers
import os
import zipfile
import shutil
import time
from datetime import datetime
import random
import pandas as pd
from django.conf import settings
from django.contrib.auth.decorators import login_required


@login_required(login_url="/accounts/login/")
def EmployeesImagesView(request, employee_id):
	
	html_template = 'portal_images.html'
	context = getDefaultContext(request)
	local_timezone = ""#settings.TIMEZONES[request.user.settings.timezone]

	employee = Employee.objects.get(id = employee_id, account_owner = request.user)

	if(request.POST):
		
		pprint(request.POST)
		
		if("new_image" in request.POST):
			files = dict(request.FILES).get('file', None)
			if(len(files) > 0):
				file = files[0]
				filename = "".join([str(random.randint(67, 90)) for c in range(10)]) + ".png"
				filepath = os.path.join(settings.IMAGE_FOLDER, filename)

				with open(filepath, 'wb+') as destination:
					for chunk in file.chunks():
						destination.write(chunk)
				
				Picture.objects.create(employee = employee, filepath = filename)

				for camera in Camera.objects.filter(account_owner = request.user):
					camera.sync()


		if("delete_image" in request.POST):
			id = int(request.POST.get("delete_image"))
			pic = Picture.objects.get(id = id)
			filepath = os.path.join(settings.IMAGE_FOLDER, pic.filepath)
			try:
				os.remove(filepath)
			except:
				pass
			pic.delete()

			for camera in Camera.objects.filter(account_owner = request.user):
				camera.sync()
			
		

		return HttpResponseRedirect(request.get_full_path())


		

	

	context["pictures"] = Picture.objects.filter(employee = employee)
	context['employee'] = employee
	context['nav_items']['Employees'][0][0] = True
	response = render(request, template_name = html_template, context=context)
	return response


