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
import pytz
import channels.layers
import os
import zipfile
import shutil
import time
from datetime import datetime
import random
import pandas as pd


def EmployeesView(request, context, local_timezone):
	html_template = 'portal_employees.html'

	if(request.POST):

		pprint(request.POST)

		if("employee_info" in request.POST):
			save_employee(request)
			return HttpResponseRedirect(request.get_full_path())
			
		if("view_employee" in request.POST):
			id = int(request.POST.get("view_employee"))
			employee = Employee.objects.get(id = id, account_owner = request.user)
			permissions = Permission.objects.filter(employee = employee)
			allowed_rooms = list(map(lambda p: p.room.id, permissions))
			context["show_employee_modal"] = True
			context["employee"] = employee
			context["allowed_rooms"] = allowed_rooms


		if("delete_employee" in request.POST):

			id = int(request.POST.get("delete_employee"))
			employee = Employee.objects.get(id = id, account_owner = request.user)
			employee.delete()
			for camera in Camera.objects.filter(account_owner = request.user):
				camera.sync()
				
			return HttpResponseRedirect(request.get_full_path())

	rooms = Room.objects.filter(account_owner = request.user)

	employees = [(employee, 
						  "",
						  Picture.objects.filter(employee = employee).count(),
						  Permission.objects.filter(employee = employee).count())
						  for employee in Employee.objects.filter(account_owner=request.user).order_by('-date_added')]
	
	context["rooms"] = rooms
	context['records'] = employees
	context['timezone'] = local_timezone

	context['nav_items']['Employees'][0][0] = True
	response = render(request, template_name = html_template, context=context)
	return response



def save_employee(request):
	
	new_name = request.POST.get("employee_name").strip()
	employee_sent = request.POST.get("employee_sent").strip()
	print("employee sent", employee_sent)
	new_employee = len(employee_sent) == 0
	
	if(new_employee):
		employee = Employee.objects.create(name = new_name, account_owner = request.user)
	else:
		employee = Employee.objects.get(name = employee_sent, account_owner = request.user)
		employee.name = new_name
		employee.save()
		Permission.objects.filter(employee = employee).delete()

	room_ids = list(filter(lambda x: "employee_checkbox_" in x, list(request.POST.keys())))
	room_ids = list(map(lambda x: int(x[18:]), room_ids))

	for room_id in room_ids:
		print(f"Permission Room: {room_id}")
		room = Room.objects.get(id = room_id)
		Permission.objects.create(employee = employee, room = room)

	#name = request.POST.get("room_name")
	#Employee.objects.create(name = name, account_owner = request.user)
