from django.db import models
from django.dispatch import receiver
from foo_auth.models import User
from django.utils import timezone
from django.conf import settings
from asgiref.sync import async_to_sync
from desktop_api.utils import getJobLogFilePath
import developer.utils.utils_cron as cron_utils
from channels.layers import get_channel_layer
import random
from desktop_api.utils import *
from developer.utils.utils import getDynamicDevices
import croniter
import cron_descriptor 
import datetime
import os
import pytz
from background_task import background
from background_task.models import Task
from pprint import pprint
from developer.templatetags.custom_tags import ProcessDateforHTML


class ApplicationVersion(models.Model):

    version = models.CharField(max_length=20, null=True, blank=True, default = "Manual")
    date_updated = models.DateTimeField(default=timezone.now)
    debug = models.BooleanField(default=False)


class Process(models.Model):
    class Meta:
        unique_together = (('process_name', 'account_owner'),)

    id = models.AutoField(primary_key=True)
    process_name = models.CharField(unique = False, max_length = 100, db_index=True, default='')
    current_version = models.CharField(unique = False, max_length = 50, default='')
    account_owner = models.ForeignKey(User, on_delete=models.CASCADE, default='')
    date_added = models.DateTimeField(default=timezone.now, blank=True)
    date_modified = models.DateTimeField(default=timezone.now, blank=True)
    process_type = models.CharField(unique=False, max_length=20, default='')
    python_version = models.CharField(unique = False, max_length = 50, default='3.8.4')
    last_run = models.DateTimeField(blank=True, null=True)


class ProcessVersion(models.Model):
    class Meta:
        unique_together = (('process_name', 'process_version'),) 

    process_name = models.ForeignKey(Process, unique=False, on_delete=models.CASCADE)
    process_version = models.CharField(unique=False, max_length=50)
    date_added = models.DateTimeField(default=timezone.now, blank=True)
    arguments = models.CharField(unique = False, max_length = 1024, default='')
    comments = models.CharField(max_length=100, default='')
    original_upload_path = models.CharField(max_length=252, default='')


class Robot(models.Model):
    class Meta:
        unique_together = (('name','account_owner'),)
    
    device_id = models.CharField(max_length=50, primary_key=True, default='')
    name = models.CharField(max_length=100, db_index=True, default='')
    creater = models.ForeignKey(User, related_name='robot_creater', unique=False, on_delete=models.SET_NULL, null=True, blank=True)
    account_owner = models.ForeignKey(User, related_name='robot_account_owner', unique=False, on_delete=models.CASCADE, null=True, blank=True)
    username = models.CharField(max_length=70, default='')
    password = models.CharField(max_length=70, default='')
    user_domain = models.CharField(max_length=70, default='')
    hostname = models.CharField(max_length=70, default='')
    os = models.CharField(max_length=70, null=True, blank=True)
    os_version = models.CharField(max_length=50, null=True, blank=True)
    last_seen = models.DateTimeField(default=timezone.now)
    date_added = models.DateTimeField(default=timezone.now)
    date_update_attempted = models.DateTimeField(default=timezone.now, null=True, blank=True)
    date_updated = models.DateTimeField(default=timezone.now)
    environment = models.CharField(max_length=15, choices = settings.ENVIRONMENTS, default = settings.ENVIRONMENTS[0][1])
    is_connected = models.BooleanField(default=False)
    is_running = models.BooleanField(default=False)
    queue = models.CharField(max_length=1000, default='')
    token = models.CharField(max_length=50, null=True, blank=True)
    last_run = models.DateTimeField(blank=True, null=True)

    uirobot_path = models.CharField( max_length=250, default="")
    uipath_log_folder = models.CharField( max_length=250, default="")

    def save(self, *args, **kwargs):
        try:
            previous_connection_status = Robot.objects.get(device_id = self.device_id).is_connected
            if(not self.is_connected and previous_connection_status):
                self.last_seen = timezone.now()
        except Exception as e:
            print("Save Warning: ", e)

        super(Robot, self).save(*args, **kwargs)


    def ping(self):
        layer = get_channel_layer()
        device_group_name = "device_" + self.device_id
        message = {"type" : "message",
                   "event" : "ping "}
        async_to_sync(layer.group_send)(device_group_name, message)


    def sendReloadSignal(self):
        try:
            layer = get_channel_layer()
            robot_group_name = "robot_" + self.account_owner.username
            message = {"type" : "message",
                    "event" : "reload"}
            async_to_sync(layer.group_send)(robot_group_name, message)
        except:
            pass


class RobotRegistrationToken(models.Model):
    token = models.CharField(max_length=50, primary_key=True)
    creater = models.ForeignKey(User, related_name='robot_registration_creater', unique=False, on_delete=models.SET_NULL, null=True, blank=True)
    account_owner = models.ForeignKey(User, related_name='robot_registration_account_owner', unique=False, on_delete=models.CASCADE, null=True, blank=True)
    used = models.BooleanField(default=False)
    date_added = models.DateTimeField(default=timezone.now)


class Job(models.Model):

    name = models.CharField(unique = False, max_length = 100, default='')
    id = models.AutoField(primary_key=True, db_index=True)
    process = models.ForeignKey(Process, related_name="job_process_name", on_delete=models.SET_NULL, blank=True, null=True)
    schedule_name = models.CharField(max_length=100, null=True, blank=True, default = "Manual")
    schedule = models.ForeignKey("Schedule", related_name="job_schedule", on_delete=models.SET_NULL, blank=True, null=True)
    device = models.ForeignKey(Robot, related_name="job_device_id", on_delete=models.SET_NULL, blank=True, null=True)
    account_owner = models.ForeignKey(User, related_name="job_account_owner", on_delete=models.CASCADE, blank=True, null=True)
    date_added = models.DateTimeField(default=timezone.now)
    date_started = models.DateTimeField(null=True, blank=True)
    date_ended = models.DateTimeField(null=True, blank=True)
    status = models.CharField(max_length=50, default = "pending")
    log_file =  models.CharField(max_length=255, default = "")
    arguments = models.CharField(unique = False, max_length = 1024, default='')
    timeout = models.IntegerField(null=True, blank=True)

    def SignalDevice(self):

        if(not self.device.is_connected):
            print("pinging device")
            self.device.ping()

        layer = get_channel_layer()
        print("Signaling device for JOB: ",self.id)
        device_group_name = "device_" + self.device.device_id
        event = "run " + str(self.id)
        message = {"type" : "message",
                   "event" : event}
        async_to_sync(layer.group_send)(device_group_name, message)


    def UpdateStatus(self):

        layer = get_channel_layer()
        local_timezone = settings.TIMEZONES[self.device.account_owner.settings.timezone]
        message = {"type": "message",
                    "event": "status_update",
                    "status": self.status,
                    "background_color": settings.JOB_STATUSES.get(self.status.lower(), ["white"])[0],
                    "job_id": self.id}

        if(self.date_ended!=None):
            date_ended = self.date_ended.astimezone(local_timezone).replace(tzinfo=None)
            message["date_ended"] = ProcessDateforHTML(date_ended)
            #.strftime(settings.TABLE_DATE_FORMAT)
            #date_ended = date_ended[:-3] + date_ended[-3].lower() + ".m."
            #message["date_ended"] = date_ended.replace(", 0", ", ").replace(". 0", ", ")

        if(self.date_started!=None):
            date_started = self.date_started.astimezone(local_timezone).replace(tzinfo=None)
            message["date_started"] = ProcessDateforHTML(date_started)
            #.strftime(settings.TABLE_DATE_FORMAT)
            #date_started = date_started[:-3] + date_started[-3].lower() + ".m."
            #message["date_started"] = date_started.replace(", 0", ", ").replace(". 0", ", ")

        job_group_name = "job_" + self.account_owner.username
        async_to_sync(layer.group_send)(job_group_name, message)
        print("Status uploaded successfully")


    def SendNewRecordMessage(self):

        layer = get_channel_layer()
        message = {"type": "message",
                   "event": "new_job_record",
                   "job_id": str(self.id),
                   "name": self.name,
                   "process": self.process.process_name,
                   "schedule": self.schedule_name,
                   "device": self.device.name,
                   "status": self.status,
                   "background_color": settings.JOB_STATUSES.get(self.status.lower(), ["white"])[0]}

        job_group_name = "job_" + self.account_owner.username
        async_to_sync(layer.group_send)(job_group_name, message)
        #print("Sending new record successfully")


    def sendReloadSignal(self):
        try:
            layer = get_channel_layer()
            robot_group_name = "job_" + self.account_owner.username
            message = {"type" : "message",
                       "event" : "reload"}
            async_to_sync(layer.group_send)(robot_group_name, message)
        except:
            pass


@receiver(models.signals.post_save, sender=Job)
def JobPostSave(sender, instance, created, *args, **kwargs):
    if (created):
        instance.SignalDevice()
        instance.SendNewRecordMessage()
        try:
            instance.process.last_run = timezone.now()
            instance.process.save()
            instance.device.last_run = timezone.now()
            instance.device.save()
        except:
            pass

        MonitorJobSignal(instance.id, creator=instance, verbose_name="MonitorJobSignal")

@background(schedule=settings.JOB_SIGNAL_DELAY_SECONDS, queue=str(random.randint(1, settings.N_BACKGROUND_WORKERS)) )
def MonitorJobSignal(job_id):
    #print(creator)
    print("Running Monitor Job at Time: ", timezone.now())
    try:
        job = Job.objects.get(id = job_id)
        job_time_limit = timezone.now() - timezone.timedelta(minutes = settings.JOB_SIGNAL_RETRY_TIME_MINUTES)
        job_timedout = job.date_added <= job_time_limit
        if(job.status.lower() == "pending"):
            if(job_timedout):
                print("Job is too old. Finishing it")
                job.status = "Device Offline"
                job.date_ended = timezone.now()
                job.save()
                job.UpdateStatus()
            else:
                job.SignalDevice()
                MonitorJobSignal(job_id, creator=job, verbose_name="MonitorJobSignal")
    except:
        pass


class ScheduleInfoOnce(models.Model):

    id = models.AutoField(primary_key=True)
    time = models.DateTimeField(null=True, blank=True)


class ScheduleInfoRepeat(models.Model):

    id = models.AutoField(primary_key=True)
    schedule_type = models.CharField(max_length=7, default="once")
    date_from = models.DateTimeField(null=True, blank=True)
    date_to = models.DateTimeField(null=True, blank=True)
    monday = models.BooleanField(default = False)
    tuesday = models.BooleanField(default = False)
    wednesday = models.BooleanField(default = False)
    thursday = models.BooleanField(default = False)
    friday = models.BooleanField(default = False)
    saturday = models.BooleanField(default = False)
    sunday = models.BooleanField(default = False)
    time_once = models.CharField(max_length=5, null=True, blank=True)
    time_from = models.CharField(max_length=5, null=True, blank=True)
    time_to = models.CharField(max_length=5, null=True, blank=True)
    repeat_type = models.CharField(max_length=8, null=True, blank=True)
    repeat_interval = models.IntegerField(null=True, blank=True)


class Schedule(models.Model):

    name = models.CharField(unique = False, max_length = 100, default='')
    id = models.AutoField(primary_key=True)
    process = models.ForeignKey(Process, related_name="schedule_process_name", on_delete=models.CASCADE, blank=True, null=True)
    account_owner = models.ForeignKey(User, related_name="schedule_account_owner", on_delete=models.CASCADE, blank=True, null=True)
    schedule_type = models.CharField(max_length=7)
    info_repeat = models.ForeignKey(ScheduleInfoRepeat, related_name="repeat_schedule_info", on_delete=models.SET_NULL, blank=True, null=True)
    info_once = models.ForeignKey(ScheduleInfoOnce, related_name="once_schedule_info", on_delete=models.SET_NULL, blank=True, null=True)
    cron_expression = models.CharField(max_length=70, blank=True, null=True)
    date_added = models.DateTimeField(default=timezone.now)
    last_run = models.DateTimeField(blank=True, null=True)
    next_run = models.DateTimeField(blank=True, null=True)
    status = models.CharField(max_length=50, default = "")
    arguments = models.CharField(unique = False, max_length = 1024, default='')
    is_active = models.BooleanField(default=True)
    description = models.CharField(max_length=70, blank=True, null=True)
    timeout = models.IntegerField(null=True, blank=True)
    dynamic_selection = models.BooleanField(default=False)
    dynamic_environment = models.CharField(max_length=20, default="Any")


    def generateScheduleDescription(self):

        if(self.schedule_type == "once"):
            description = "Just Once"  #.format(self.info_once.time)

        if(self.schedule_type == "repeat"):
            if(self.info_repeat.schedule_type=="once"):
                description = "Once a day at {}".format(self.info_repeat.time_once)
            else:
                time_from = self.info_repeat.time_from if self.info_repeat.time_from!=None else "00:00"
                time_to = self.info_repeat.time_to if self.info_repeat.time_to!=None else "23:59"
                repeat_interval = self.info_repeat.repeat_interval
                repeat_type = self.info_repeat.repeat_type
                if(int(repeat_interval)<2):
                    repeat_type = repeat_type[:-1]
                description = "Every {} {} between {} and {}".format(repeat_interval, repeat_type, time_from, time_to)

        if(self.schedule_type == "custom"):
            description = cron_descriptor.get_description(self.cron_expression)

        self.description = description[:70]
        print("Description: {}".format(description))
        return description

    def getNextRun(self):
        #print()
        #print("**** GET NEXT RUN ****")
        if(not self.is_active):
            self.next_run = None
            return self.next_run

        local_timezone = settings.TIMEZONES[self.account_owner.settings.timezone]

        if(self.schedule_type == "once"):
            next_run = self.getNextRunOnce(local_timezone)
        if(self.schedule_type == "repeat"):
            next_run = self.getNextRunRepeat(local_timezone)
        if(self.schedule_type == "custom"):
            next_run = self.getNextRunCustom(local_timezone)

        if(next_run!=None and self.schedule_type != "once"):
            next_run = pytz.timezone('utc').localize(next_run)
        self.next_run = next_run
        self.generateScheduleDescription()

        return next_run

    def getNextRunOnce(self, local_timezone):
        print("Getting Next Run")
        utc_timezone = pytz.timezone('utc')
        datetime_once = self.info_once.time
        try:
            datetime_once = utc_timezone.localize(datetime_once).astimezone(utc_timezone)#.replace(tzinfo=None)
        except:
            pass
        print(datetime_once)
        next_run = datetime_once if datetime_once > timezone.now() else None
        print("Next run: ", next_run)
        try: datetime_once = datetime_once.replace(tzinfo=None)
        except: pass
        return next_run

    def getNextRunRepeat(self, local_timezone):

        week_string = ""
        if(self.info_repeat.monday): week_string += "MON,"
        if(self.info_repeat.tuesday): week_string += "TUE,"
        if(self.info_repeat.wednesday): week_string += "WED,"
        if(self.info_repeat.thursday): week_string += "THU,"
        if(self.info_repeat.friday): week_string += "FRI,"
        if(self.info_repeat.saturday): week_string += "SAT,"
        if(self.info_repeat.sunday): week_string += "SUN,"
        if(week_string[-1]==","):
            week_string = week_string[:-1]
        if(week_string==""):
            week_string = "*"

        current_time_local = timezone.now().astimezone(local_timezone).replace(tzinfo=None)
        if(self.info_repeat.schedule_type == "once"):
            try:
                hour, minute = self.info_repeat.time_once.split(":")
                hour = int(hour)
                minute = int(minute)
                cron_expression = "{} {} * * {}".format(minute, 
                                                        hour, 
                                                        week_string)
            except:
                return None

        if(self.info_repeat.schedule_type == "repeat"):

            if(self.info_repeat.time_from!=None and ":" in self.info_repeat.time_from):
                start_hour, start_minute = self.info_repeat.time_from.split(":")
                start_hour = int(start_hour)
                start_minute = int(start_minute)
            else:
                start_hour = 0
                start_minute = 0

            if(self.info_repeat.time_to!=None and ":" in self.info_repeat.time_to):
                end_hour, end_minute = self.info_repeat.time_to.split(":")
                end_hour = int(end_hour)
                end_minute = int(end_minute)
            else:
                end_hour = 23
                end_minute = 59

            if(self.info_repeat.repeat_type =="hours" ):
                cron_expression = "{} {}-{}/{} * * {}".format(start_minute, 
                                                              start_hour, 
                                                              end_hour, 
                                                              self.info_repeat.repeat_interval, 
                                                              week_string)
            elif(self.info_repeat.repeat_type =="minutes" ):
                cron_expression = "0/{} {}-{}/1 * * {}".format(self.info_repeat.repeat_interval,
                                                               start_hour,
                                                               end_hour, 
                                                               week_string)

            else:
                return None
                

        #print("Cron expression generated: {}".format(cron_expression))
        #print("Current_time: {}".format(current_time_local))

        if(self.info_repeat.time_from != None):
            time_from = datetime.datetime.strptime(self.info_repeat.time_from, "%H:%M").time()
        else:
            time_from = datetime.time(hour = 0, minute = 0 )
        if(self.info_repeat.time_to != None):
            time_to = datetime.datetime.strptime(self.info_repeat.time_to,"%H:%M").time()
        else:
            time_to = datetime.time(hour = 23, minute = 59)
        
        utc_timezone = pytz.timezone('utc')
        if(self.info_repeat.date_from!=None):
            datetime_from = self.info_repeat.date_from
            try: utc_timezone.localize(datetime_from)
            except: pass
            datetime_from = datetime_from.astimezone(local_timezone).replace(tzinfo=None)
            datetime_from = datetime.datetime.combine(datetime_from, time_from)
        else:
            datetime_from = current_time_local

        if(self.info_repeat.date_to!=None):
            datetime_to = self.info_repeat.date_to
            try: utc_timezone.localize(datetime_to)
            except: pass
            datetime_to = datetime_to.astimezone(local_timezone).replace(tzinfo=None)
            datetime_to = datetime.datetime.combine(datetime_to, time_to)
        else:
            datetime_to = datetime.datetime(day=31, month=12, year=9999, hour=23, minute=59)


        #print("Time From: ", time_from, type(time_from))
        #print("Time To: ", time_to, type(time_to))
        #print("DateTime From: ", datetime_from, type(datetime_from))
        #print("DateTime To: ", datetime_to, type(datetime_to))

        if(time_from > time_to or datetime_from > datetime_to or current_time_local > datetime_to):
            return None

        cron = croniter.croniter(cron_expression, current_time_local)
        while(True):
            next_run = cron.get_next(datetime.datetime)
            next_run_time = next_run.time()
            if(next_run > datetime_to):
                return None
            if( time_from <= next_run_time <= time_to  and  datetime_from <= next_run <= datetime_to):
                break

        #print("Next Run: {}".format(next_run))
        next_run = local_timezone.localize(next_run).astimezone(tz=pytz.timezone('utc')).replace(tzinfo=None)
        #print("Next Run (UTC): {}".format(next_run))

        return next_run

    def getNextRunCustom(self, local_timezone):
        current_time_local = timezone.now().astimezone(local_timezone).replace(tzinfo=None)
        try:
            cron = croniter.croniter(self.cron_expression, current_time_local)
            next_run = cron.get_next(datetime.datetime)
            next_run = local_timezone.localize(next_run).astimezone(tz=pytz.timezone('utc')).replace(tzinfo=None)
        except:# Exception as e:
            #print(e)
            next_run = None
        return next_run

    def CreateScheduleJob(self):
        print("Creating schdeule jobs")
        #print()
        #print("******* We are in CREATESCHEDULEJOB *********")
        #print()
        layer = get_channel_layer()
        local_timezone = settings.TIMEZONES[self.account_owner.settings.timezone]
        job_time_limit = timezone.now() - timezone.timedelta(hours = 12)
        previous_jobs = Job.objects.filter(process=self.process, date_added__gte = job_time_limit, account_owner=self.account_owner)
        devices_still_running_process = [job.device.name for job in previous_jobs if job.status.lower() not in settings.ENDING_STATUSES]
 
        #print("Schedule Devices: ", schedule_devices)
        now = timezone.now().astimezone(local_timezone).replace(tzinfo=None)
        job_name = "{}_{}.{}.{}_{}.{}.{}".format(self.process.process_name, now.day, now.month,now.year, now.hour, now.minute, now.second).replace(" ", ".")
        atleast_one_job_created = False
        schedule_devices = ScheduleDevices.objects.filter(schedule=self)
        print("Devices still running process:  ", devices_still_running_process)

        

        if(self.dynamic_selection):

            devices = getDynamicDevices(self.account_owner, self.dynamic_environment, n=1)
            for device in devices:
                job = Job(name=job_name[:100],
                        process = self.process, 
                        schedule_name = self.name[:100], 
                        schedule = self,
                        device = device,
                        account_owner = self.account_owner,
                        arguments = self.arguments,
                        timeout=self.timeout)
                job.save()
                job.log_file = getJobLogFilePath(job)
                job.save()
                atleast_one_job_created = True

        else:

            for schedule_device in schedule_devices:
                try:
                    device = schedule_device.device
                    no_previous_jobs_on_device = device.name not in devices_still_running_process
                    if(no_previous_jobs_on_device):
                        print("Creating Job for schedule '{}'".format(self.name))
                        job = Job(name=job_name[:100],
                                    process = self.process, 
                                    schedule_name = self.name[:100], 
                                    schedule = self,
                                    device = device,
                                    account_owner = self.account_owner,
                                    arguments = self.arguments,
                                    timeout=self.timeout)
                        job.save()
                        job.log_file = getJobLogFilePath(job)
                        job.save()
                        atleast_one_job_created = True
                    else:
                        #print("We are here somehow")
                        message = { "type": "message",
                                    "event": "notification",
                                    "notification_type": "danger",
                                    "message": "Already a pending job for process '{}' at device '{}'".format(self.process.process_name, device.name)}
                        webuser_group_name = "webuser_"+self.account_owner.username
                        async_to_sync(layer.group_send)(webuser_group_name, message)
        
                except:# Exception as e:
                    #print("ERROR IN TRIGGER SCHEDULE:  ", e)
                    pass
            

        if(atleast_one_job_created):
            #job.sendReloadSignal()
            message = { "type": "message",
                        "event": "notification",
                        "notification_type": "info",
                        "message": "{} triggered".format(self.name)}
            webuser_group_name = "webuser_"+self.account_owner.username
            async_to_sync(layer.group_send)(webuser_group_name, message)

        return atleast_one_job_created

    def CommunicateScheduleRunTimes(self):
        print("Communicating schedule runtimes")
        layer = get_channel_layer()
        local_timezone = settings.TIMEZONES[self.account_owner.settings.timezone]

        send = False
        message = {"type": "message",
                    "event": "schedule_runtimes_update",
                    "schedule_id": self.id}
        
        if(self.next_run!=None):
            next_run = self.next_run.astimezone(local_timezone).replace(tzinfo=None)
            message["next_run"] = ProcessDateforHTML(next_run)
            send = True
        else:
            message["next_run"] = ""

        if(self.last_run!=None):
            last_job_date = self.last_run.astimezone(local_timezone).replace(tzinfo=None)
            message["last_run"] = ProcessDateforHTML(last_job_date)
            send = True
        else:
            message["last_run"] = ""

        if(send):
            job_group_name = "schedule_"+self.account_owner.username
            async_to_sync(layer.group_send)(job_group_name, message)

    def ReconfigureSchedule(self):
        print("Reconfiguring Schedule {} ({})".format(self.name, self.id))
        current_schedule_tasks = Task.objects.filter(creator_object_id = self.id)
        print("Number of tasks found {} ({}): ".format(self.name, self.id), current_schedule_tasks.count())
        current_schedule_tasks.delete()
        self.generateScheduleDescription()

@receiver(models.signals.post_save, sender=Schedule)
def SchedulePostSave(sender, instance, created, *args, **kwargs):
    if(instance.next_run!=None):
        TriggerSchedule(instance.id, schedule=instance.next_run, creator=instance, verbose_name="ScheduleNextRun")


@background(queue=str(random.randint(1, settings.N_BACKGROUND_WORKERS)))
def TriggerSchedule(schedule_id):
    print("Running Schedule Trigger at Time: ", timezone.now())
    try:
        schedule = Schedule.objects.get(id = schedule_id)
        atleast_one_job_created = schedule.CreateScheduleJob()
        if(atleast_one_job_created):
            schedule.last_run = timezone.now()
        schedule.getNextRun()
        schedule.CommunicateScheduleRunTimes()
        schedule.save()

    except Exception as e:
        print("Error (schedule_id: {}) : {}".format(schedule_id, e))
        pass


class ScheduleDevices(models.Model):

    id = models.AutoField(primary_key=True)
    schedule = models.ForeignKey(Schedule, related_name="schedule_name1", on_delete=models.CASCADE, null=True, blank=True)
    device = models.ForeignKey(Robot, related_name="schedule_device1", on_delete=models.CASCADE, null=True, blank=True)

