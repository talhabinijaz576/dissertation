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


