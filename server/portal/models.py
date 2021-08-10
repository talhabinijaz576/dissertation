from django.db import models
from django.dispatch import receiver
from foo_auth.models import User
from django.utils import timezone
from django.conf import settings
from asgiref.sync import async_to_sync
from channels.layers import get_channel_layer
import cron_descriptor 
import datetime
import os
import pytz
from background_task import background
from background_task.models import Task
from pprint import pprint
from portal.templatetags.custom_tags import ProcessDateforHTML
import time
import random


def generate_code():
    code = "".join([chr(random.randint(65, 90)) for _ in range(5)])
    return code



class Room(models.Model):

    id = models.AutoField(primary_key=True)
    name = models.CharField(unique = False, max_length = 100, db_index=True, default='')
    date_added = models.DateTimeField(default=timezone.now, blank=True)
    account_owner = models.ForeignKey(User, on_delete=models.CASCADE, default='')
    code = models.CharField(unique = False, max_length = 100, db_index=True, default = generate_code)


    

class Camera(models.Model):

    id = models.AutoField(primary_key=True)
    name = models.CharField(unique = False, max_length = 100, db_index=True, default='')
    account_owner = models.ForeignKey(User, on_delete=models.CASCADE, default='')
    date_added = models.DateTimeField(default=timezone.now, blank=True)
    date_synced = models.DateTimeField(default=timezone.now, blank=True)
    room = models.ForeignKey(Room, on_delete=models.CASCADE, default='')
    is_connected = models.BooleanField(default=False)
    token = models.CharField(unique = False, max_length = 100, db_index=True, default='')


    def on_connect(self):
        try:
            print("on_connect")
            self.send_connection_message(True)
            self.is_connected = True
            self.save()
            print("saved")
        except:
            pass


    def on_disconnect(self):
        try:
            print("on_disconnect")
            self.send_connection_message(False)
            self.is_connected = False
            self.save()
            print("saved")
        except:
            pass


    def send_connection_message(self, is_connected):
        
        try:
            layer = get_channel_layer()
            message = {"type": "message",
                        "event": "connection_update",
                        "status": is_connected,
                        "camera_name": self.name,
                        "camera_id": str(self.id)}

            group_name = "portal_" + self.account_owner.username
            print(f"Sending connection message ({is_connected}) to {group_name}")
            async_to_sync(layer.group_send)(group_name, message)
        except:
            pass


    def sync(self):
        
        try:
            layer = get_channel_layer()
            message = {"type": "message",
                        "event": "sync"}

            group_name = "device_" + self.token
            print(f"Sending resync message to {group_name}")
            async_to_sync(layer.group_send)(group_name, message)
        except:
            pass


    def check_access(self, employee):

        has_access = Permission.objects.filter(employee = employee, room = self.room).exists()
        print(f"Checking access for {employee.name} (Access: {has_access})")

        return has_access


    def raise_event(self, detected, has_access, name):

        if(detected):

            if(has_access):
                message = "Access granted"
            else:
                message = "Unauthorized person access request denied"

            event = Event.objects.create(message = message, 
                                         subject = name, 
                                         access_granted = has_access, 
                                         room = self.room.name,
                                         camera = self.name,
                                         account_owner = self.account_owner)

        else:
            message = "Unknown person access request denied"
            event = Event.objects.create(message = message, 
                                         subject = name, 
                                         access_granted = has_access, 
                                         room = self.room.name,
                                         camera = self.name,
                                         account_owner = self.account_owner)

        if(not detected or not has_access):
            try:
                layer = get_channel_layer()
                message = {"type": "message",
                            "event": "alert",
                            "alert": f"{message} at {self.room.name}" }

                group_name = "portal_" + self.account_owner.username
                print(f"Sending alert ({message}) to {group_name}")
                async_to_sync(layer.group_send)(group_name, message)
            except:
                pass

        return event




class Employee(models.Model):

    id = models.AutoField(primary_key=True)
    name = models.CharField(unique = False, max_length = 100, db_index=True, default='')
    date_added = models.DateTimeField(default=timezone.now, blank=True)
    date_synced = models.DateTimeField(default=timezone.now, blank=True)
    account_owner = models.ForeignKey(User, on_delete=models.CASCADE, default='')


    

class Permission(models.Model):

    id = models.AutoField(primary_key=True)
    room = models.ForeignKey(Room, on_delete=models.CASCADE, default='')
    employee = models.ForeignKey(Employee, on_delete=models.CASCADE, default='')



class Picture(models.Model):

    id = models.AutoField(primary_key=True)
    employee = models.ForeignKey(Employee, on_delete=models.CASCADE, default='')
    filepath = models.CharField(unique = False, max_length = 100, db_index=True, default='')





class Event(models.Model):


    id = models.AutoField(primary_key=True)
    subject = models.CharField(unique = False, max_length = 100, db_index=True, default='')
    date = models.DateTimeField(default=timezone.now, blank=True)
    message = models.CharField(unique = False, max_length = 100, db_index=True, default='')
    access_granted = models.BooleanField(default=False)
    read = models.BooleanField(default=False)
    room = models.CharField(unique = False, max_length = 100, db_index=True, default='')
    camera = models.CharField(unique = False, max_length = 100, db_index=True, default='')
    account_owner = models.ForeignKey(User, on_delete=models.CASCADE)









#@receiver(models.signals.post_save, sender=Event)
#3def EventPostSave(sender, instance, created, *args, **kwargs):
#    if (created):
#        instance.SetTranscriptFile()




