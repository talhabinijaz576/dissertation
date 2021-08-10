from django.conf import settings
from django.utils import timezone
from rest_framework.response import Response
from django.http import HttpResponse, HttpResponseNotFound
import os


def returnFile(file_path):
	try:
		fsock = open(file_path,"rb")
		file_name = os.path.basename(file_path)
		response = HttpResponse(fsock)
		response['Content-Disposition'] = 'attachment; filename=' + file_name            
	except IOError:
		response = HttpResponseNotFound()
	return response
	

def getSystemLogFilePath(device, day):
	path = os.path.join(settings.LOG_ROOT, device.account_owner.username)
	if(not os.path.isdir(path)): 
		os.mkdir(path)
	path = os.path.join(path, "SystemLogs")
	if(not os.path.isdir(path)): 
		os.mkdir(path)
	path = os.path.join(path, device.device_id)
	if(not os.path.isdir(path)): 
		os.mkdir(path)
	path = os.path.join(path, str(day.strftime("%B %Y")))
	if(not os.path.isdir(path)): 
		os.mkdir(path)
	filename = day.strftime("%d-%m-%Y") + ".txt"
	path = os.path.join(path, filename)
	return path



def getJobLogFilePath(job):
	today = timezone.now()
	process_folder = "ProcessLogs"
	path = os.path.join(settings.LOG_ROOT, job.device.account_owner.username)
	if(not os.path.isdir(path)): 
		os.mkdir(path)
	path = os.path.join(path, process_folder)
	if(not os.path.isdir(path)): 
		os.mkdir(path)
	path = os.path.join(path, job.process.process_name)
	if(not os.path.isdir(path)): 
		os.mkdir(path)
	path = os.path.join(path, str(today.strftime("%B %Y")))
	if(not os.path.isdir(path)): 
		os.mkdir(path)
	filename = job.process.process_name +"({})".format(job.id)+ ".txt"
	path = os.path.join(path, filename)
	relative_path = os.path.relpath(path, settings.LOG_ROOT)
	return relative_path