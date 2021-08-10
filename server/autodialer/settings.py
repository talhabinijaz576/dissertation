"""
Django settings for dialer project.

Generated by 'django-admin startproject' using Django 3.0.7.

For more information on this file, see
https://docs.djangoproject.com/en/3.0/topics/settings/

For the full list of settings and their values, see
https://docs.djangoproject.com/en/3.0/ref/settings/
"""

import os
from . import timezones
import platform
from country_list import countries_for_language
import random



# Random

RAND = "00000011"


# Build paths inside the project like this: os.path.join(BASE_DIR, ...)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/3.0/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = '20w3#lk-b+u30kedm6#=dhip@h-(hv)(a2y&3b3a1@d++c)o*r'

# SECURITY WARNING: don't run with debug turned on in production!
IS_LINUX = platform.system() == "Linux"

#DEBUG = not IS_LINUX
DEBUG = True


ALLOWED_HOSTS = ['*']



# Application definition

INSTALLED_APPS = [
   # 'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'background_task',
    'channels',
    'foo_auth',
    'home',
    'portal',
    'login_app',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'autodialer.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [os.path.join(BASE_DIR, 'templates')], 
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'autodialer.wsgi.application'
ASGI_APPLICATION = 'autodialer.routing.application'

redis_host = os.environ.get('REDIS_HOST', 'localhost')

if(IS_LINUX):

    CHANNEL_LAYERS = {
        'default': {
            'BACKEND': 'channels_redis.core.RedisChannelLayer',
            'CONFIG': {
                'hosts': [(redis_host, 6379)],
                #'capacity': 1500,
                #'expiry': 10,
            },
        }
    }

else:

    CHANNEL_LAYERS = {
    "default": {
        "BACKEND": "channels.layers.InMemoryChannelLayer"
   }
}


# Database
# https://docs.djangoproject.com/en/3.0/ref/settings/#databases

if (IS_LINUX):

    OLD_DATABASES = {
        'default': {
            'ENGINE': 'django.db.backends.postgresql_psycopg2',
            'NAME': 'dialer',
            'USER': 'orc_user',
            'PASSWORD': 'Jazee576',
            'HOST': 'localhost',
            'PORT': '',
        }
    }

    DATABASES = {
        'default': {
            'ENGINE': 'django.db.backends.postgresql_psycopg2',
            'NAME': 'defaultdb',
            'USER': 'doadmin',
            'PASSWORD': 'fskcmkavawday0td',
            'HOST': 'db-postgresql-ams3-73046-do-user-8348754-0.b.db.ondigitalocean.com',
            'PORT': '25060',
        }
    }
    
else:

    DATABASES = {
        'default': {
            'ENGINE': 'django.db.backends.sqlite3',
            'NAME': os.path.join(BASE_DIR, 'db.sqlite3'),
        }
    }





# Password validation
# https://docs.djangoproject.com/en/3.0/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]

# Internationalization
# https://docs.djangoproject.com/en/3.0/topics/i18n/

LANGUAGE_CODE = 'en-us'

TIME_ZONE = 'UTC'

USE_I18N = True

USE_L10N = True

USE_TZ = True

DEFAULT_PROJECT_TIMEZONE = "Coordinated Universal Time UTC+00"
TIMEZONES = timezones.TimeZoneInfo()

DEFAULT_DATE_FORMAT = "dd/mm/yy"
DATE_FORMATS = ["dd/mm/yy", "mm/dd/yy", "yy/mm/dd"]

TABLE_DATE_FORMAT = "%b. %d, %Y, %I:%M %p."

RECORD_PER_PAGE = 9



IMAGE_FOLDER = os.path.join(BASE_DIR, "static/pictures")
if(not os.path.exists(IMAGE_FOLDER)):
    os.mkdir(IMAGE_FOLDER)





STATUSES = {"pending" : "silver", 
          "calling" : "silver",
          "in progress" : "blue", 
          "completed" : "green",
          "declined" : "red",
          "cancelled" : "bLack",
          "stopped" : "bLack",
          }



# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/3.0/howto/static-files/

#STATIC_ROOT = os.path.join(BASE_DIR, 'static', 'dialer', 'css')
#STATIC_ROOT = os.path.join(BASE_DIR, 'static')

STATICFILES_DIRS = ( os.path.join(BASE_DIR, 'static'),)
STATIC_URL = '/static/'
#STATIC_URL = '/Users/talhaijaz/Desktop/personal_projects/rpa_platform/dialer/static/'

AUTH_USER_MODEL = 'foo_auth.User'

LOGIN_REDIRECT_URL = '/'
LOGOUT_REDIRECT_URL = '/'



COUNTRIES = list(dict(countries_for_language('en')).values())