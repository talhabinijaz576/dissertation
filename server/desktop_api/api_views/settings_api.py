from rest_framework.response import Response
from django.http import HttpResponse, HttpResponseNotFound
from rest_framework import views
from desktop_api.permissions import AuthenticateDevice
from asgiref.sync import async_to_sync, sync_to_async 
from channels.layers import get_channel_layer
from dialer.models import Robot, Job, Process, ApplicationVersion, ProcessVersion
from foo_auth.models import ProjectConfig
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



class settingsUpdateView(views.APIView):

    permissions_classes = (AuthenticateDevice, )

    def post(self, request):

        body = request.body
        pprint(request)
        try:
            body = ast.literal_eval(body.decode("UTF-8"))
        except Exception as e:
            body = {}

        pprint(body)

        response = {"success": True}
        verified, response, device =  self.verify(body, response)
        response["success"] = verified

        if(verified):
            #local_timezone = settings.TIMEZONES[device.account_owner.settings.timezone]
            setting_type = body.get("setting_type", "device").lower()
            setting_to_change = body.get("setting_name", "NONE")
            setting_value = body.get("setting_value")#.replace("\\", "\\\\")
            try:
                if(setting_type == "project"):
                    ProjectConfig.objects.filter(id = device.account_owner.settings.id).update(**{setting_to_change: setting_value})

                elif(setting_type == "device"):
                    Robot.objects.filter(device_id = device.device_id).update(**{setting_to_change: setting_value})
                
                #device.account_owner.settings.save()
            except Exception as e:
                response["success"] = False
                response["Message"] = str(e)


        return Response(response)



    def verify(self, body, response):
        verified = True
        device = None
        try:
            device_id = body["device_id"]
            device = Robot.objects.get(device_id=device_id)
            verified = True

        except Exception as e:
            response["message"] = "Invalid Device"
            verified = False

        return verified, response, device