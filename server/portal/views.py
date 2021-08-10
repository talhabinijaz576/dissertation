from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.conf import settings
from django.contrib.auth.decorators import login_required
from portal.site_views.view_rooms import RoomsView, CamerasView
from portal.site_views.view_history import HistoryView, ReportsView, AlertsView
from portal.site_views.view_employees import EmployeesView
from pprint import pprint
import os
import zipfile
import shutil
import pytz



@login_required(login_url="/accounts/login/")
def index(request):
	if(request.user.user_type=="USER"):
		return PortalHomeView(request)
	else:
		return redirect("/")


def getDefaultContext(request):
	context = {"username": request.user.username,
			   "project_name": request.user.username,
			   "errors": [],
			   "messages": [],
			   "nav_items": {},
			   "RAND": settings.RAND}


	context["primary_account"] = True
	

	context['nav_items']["Rooms"] = [[False, "/portal/rooms", "meeting_room"]]
	context['nav_items']["Cameras"] = [[False, "/portal/cameras", "videocam"]]
	context['nav_items']["Employees"] = [[False, "/portal/employees", "people"]]
	context['nav_items']["Alerts"] = [[False, "/portal/alerts", "warning"]]
	context['nav_items']["Authorizations"] = [[False, "/portal/history", "history"]]
	context['nav_items']["Reports"] = [[False, "/portal/reports", "summarize"]]
	


	return context


def PortalHomeView(request):

	context = getDefaultContext(request)
	local_timezone = settings.TIMEZONES[request.user.settings.timezone]
	page = request.path.replace("portal", "").replace("/", '')

	#print("Page:",page)

	if('rooms' in page): 
		section_name = "dashboard"
		context["section_name"] = section_name
		view = RoomsView(request, context, local_timezone)

	elif('cameras' in page):
		view = CamerasView(request, context, local_timezone)

	elif('employees' in page):
		view = EmployeesView(request, context, local_timezone)

	elif('alerts' in page):
		view = AlertsView(request, context, local_timezone)

	elif('reports' in page):
		view = ReportsView(request, context, local_timezone)

	elif('history' in page):
		view = HistoryView(request, context, local_timezone)

	else:
		view = RoomsView(request, context, local_timezone)

	return view

