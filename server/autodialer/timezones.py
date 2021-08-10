import pytz
from pprint import pprint
from . import countryinfo
import datetime

class TimeZoneInfo:

    def __init__(self):
        timezones = countryinfo.countries
        self.timezones = {}
        self.list = []
        for timezone in timezones:
            for tz_str in timezone["timezones"]:
                tz = pytz.timezone(tz_str)
                offset = self.getUTCOffset(tz)
                name = timezone["name"]+"/"+tz_str.split(r"/")[1]
                self.timezones[name] = tz
                name = name+" "+offset 
                #self.timezones[name] = tz
                self.list.append(name)
        self.list.sort()

        utc_str = "Coordinated Universal Time"
        self.timezones[utc_str] = pytz.timezone("UTC")
        utc_str = "Coordinated Universal Time UTC+00"
        self.list.insert(0, utc_str)

        #pprint(self.timezones)


    def getUTCOffset(self, timezone):
        dt = datetime.datetime.utcnow()
        try:
            offset = timezone.utcoffset(dt)
        except:
            offset = timezone.utcoffset(dt, is_dst=True)
        offset_seconds = offset.seconds
        offset_days = offset.days
        offset_hours = ( offset_seconds / 3600.0 ) + (offset_days*24.0)
        offset_minutes = int((offset_hours % 1) * 60)
        
        offset_hours = int(offset_hours)
        sign = "+" if offset_days >=0 else ""
        offset = "UTC{}{:01d}:{:02d}".format(sign, offset_hours, offset_minutes)
        return offset

    def __getitem__(self, key):
        key = key.split(" UTC")[0].strip()
        local_timezone = self.timezones.get(key, pytz.timezone('utc'))
        #print("GETITEM: {} {}".format(key, local_timezone))

        return local_timezone


