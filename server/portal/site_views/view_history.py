from django.shortcuts import render, redirect
from django.http import HttpResponse, HttpResponseRedirect, HttpResponseNotFound
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
from portal.models import Employee, Room, Camera, Event
import pytz
import channels.layers
import os
import zipfile
import shutil
import time
from datetime import datetime
import random
import pandas as pd
from pprint import pprint



@login_required(login_url="/accounts/login/")
def HistoryView(request, context, local_timezone):
	html_template = 'portal_history.html'

	if(request.POST):

		return HttpResponseRedirect(request.get_full_path())


	events = Event.objects.filter(account_owner = request.user, access_granted = True).order_by('-date')

	context['records'] = events
	context['timezone'] = local_timezone

	context['title'] = "Authorizations"

	context['nav_items']['Authorizations'][0][0] = True
	response = render(request, template_name = html_template, context=context)
	return response


@login_required(login_url="/accounts/login/")
def AlertsView(request, context, local_timezone):
	
	html_template = 'portal_history.html'

	if(request.POST):

		return HttpResponseRedirect(request.get_full_path())


	events = Event.objects.filter(account_owner = request.user, access_granted = False).order_by('-date')

	context['records'] = events
	context['timezone'] = local_timezone

	context['title'] = "Alerts"

	context['nav_items']['Alerts'][0][0] = True
	response = render(request, template_name = html_template, context=context)
	return response


def returnFile(filepath):
	try:
		fsock =  open(filepath,"rb")
		response = HttpResponse(fsock)

		file_name = os.path.basename(filepath)
		response['Content-Disposition'] = 'attachment; filename=' + file_name            
	except IOError:
		response = HttpResponseNotFound()
	return response


@login_required(login_url="/accounts/login/")
def ReportsView(request, context, local_timezone):
	html_template = 'portal_reports.html'

	if(request.POST):

		pprint(request.POST)

		if("employee_report" in request.POST):
			id = request.POST.get("employee_report")
			employees = Employee.objects.filter(id = id, account_owner = request.user)
			if(employees.exists()):

			    employee = employees[0]
			    events = Event.objects.filter(subject = employee.name).order_by('-date')

			    def get_info(event):
			        row = f"{event.date};{event.subject};{event.room};{event.camera};{event.access_granted};{event.message}"
			        return row

			    lines = ["Date;Person;Room;Camera;Result;Message"] + list(map(get_info, events))
			    lines = list(map(lambda x: x.replace(";", ";") + "\n", lines))
			    pprint(lines)
                
			    filename = f"report_{employee.name}.csv"

			    with open(filename, "w") as f:
			        f.writelines(lines)

			    return returnFile(filename)

		return HttpResponseRedirect(request.get_full_path())


	context['employees'] = Employee.objects.filter(account_owner = request.user)
	context['timezone'] = local_timezone

	context['nav_items']['Reports'][0][0] = True
	response = render(request, template_name = html_template, context=context)
	return response






	