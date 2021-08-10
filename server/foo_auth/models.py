from django.contrib.auth.models import (
    AbstractBaseUser, BaseUserManager, PermissionsMixin
)
from django.db import models
from django.conf import settings
from django.utils import timezone
from django.utils.translation import gettext_lazy as _



class ProjectUsage(models.Model):
    id = models.AutoField(primary_key=True)
    primary_username = models.CharField(_('primary_username'), max_length=50, null=True, blank=True)
    n_installer_downloaded = models.IntegerField(_('n_installer_downloaded'), default=0)
    last_installer_download_date = models.DateTimeField(blank=True, null=True)



class ProjectConfig(models.Model):
    id = models.AutoField(primary_key=True)
    timezone = models.CharField(max_length=75, default=settings.DEFAULT_PROJECT_TIMEZONE)
    date_format = models.CharField(max_length=75, default=settings.DEFAULT_DATE_FORMAT)
    is_debug = models.BooleanField(_("is_debug"), default = False)
    is_demo_account = models.BooleanField(_("is_demo_account"), default = False)
    is_verified = models.BooleanField(_("is_verified"), default = False)

    primary_username = models.CharField(_('primary_username'), max_length=50, null=True, blank=True)
    name = models.CharField(_('name'), max_length=50, null=True, blank=True)
    phone = models.CharField(_('phone'), max_length=15, null=True, blank=True)
    company_name = models.CharField(_('company name'), max_length=50, null=True, blank=True)
    country = models.CharField(_('country'), max_length=50, null=True, blank=True)
    company_size = models.CharField(_('company size'), max_length=30, null=True, blank=True)
    business_role = models.CharField(_('business_role'), max_length=50, null=True, blank=True)

    #n_installer_downloaded = models.IntegerField(_('n_installer_downloaded'), default=0)




class UserManager(BaseUserManager):
    def create_user(
            self, username="", email="", first_name="", last_name="", password="", company_name="", user_type="NONE", account_owner = None, is_primary_user=False, is_project_admin=False, commit=True):

        user = self.model(
        	username=username,
            email=self.normalize_email(email),
            first_name=first_name,
            last_name=last_name,
            user_type=user_type,
            account_owner = account_owner,
            is_primary_user = is_primary_user,
            is_project_admin = is_project_admin,
        )
        if(user_type=="PROJECT"):
            project_settings = ProjectConfig.objects.create(company_name = company_name)
            user.settings = project_settings
            user.usage = ProjectUsage.objects.create()

        user.set_password(password)
        if commit:
            user.save(using=self._db)
        elif(user_type=="PROJECT"):
            project_settings.delete()
            
        return user

    def create_superuser(self,username, password, first_name="", last_name="" ):

        user = self.create_user(
        	username = username,
            password=password,
            first_name=first_name,
            last_name=last_name,
            commit=False,
        )
        user.is_staff = True
        user.is_superuser = True
        user.save(using=self._db)
        return user


class User(AbstractBaseUser, PermissionsMixin):
    username = models.CharField(_('username'), max_length=50, unique = True, primary_key=True)
    email = models.EmailField(verbose_name=_('email address'), max_length=255, blank=True)

    first_name = models.CharField(_('first name'), max_length=30, blank=True)
    last_name = models.CharField(_('last name'), max_length=150, blank=True)
    user_type = models.CharField(_('user type'), max_length=10, blank=True, default="NONE")
    is_active = models.BooleanField(_('active'), default=True)
    is_staff = models.BooleanField(_('staff status'), default=False)
    is_primary_user = models.BooleanField(_('primary_user'), default=False)
    is_project_admin = models.BooleanField(_('project_admin'), default=False)
    account_owner = models.ForeignKey("self", on_delete = models.CASCADE, blank=True, null=True)
    settings =  models.ForeignKey(ProjectConfig, related_name="project_settings", on_delete = models.CASCADE, blank=True, null=True)
    usage =  models.ForeignKey(ProjectUsage, related_name="project_usage", on_delete = models.CASCADE, blank=True, null=True)
    # is_superuser field provided by PermissionsMixin
    # groups field provided by PermissionsMixin
    # user_permissions field provided by PermissionsMixin

    date_joined = models.DateTimeField(
        _('date joined'), default=timezone.now
    )

    objects = UserManager()

    USERNAME_FIELD = "username"
    REQUIRED_FIELDS = []#'first_name', 'last_name']

    def ConfigureUsageData(self):
        if(self.usage==None):
            try:
                primary_username = self.settings.primary_username
            except:
                primary_username = ""
            self.usage = ProjectUsage.objects.create(primary_username = primary_username)
            self.save()

    def get_full_name(self):

        full_name = str(self.username)
        return full_name.strip()

    def __str__(self):
        return '{} <{}>'.format(self.get_full_name(), self.email)

    def has_perm(self, perm, obj=None):
        "Does the user have a specific permission?"
        # Simplest possible answer: Yes, always
        return True

    def has_module_perms(self, app_label):
        "Does the user have permissions to view the app `app_label`?"
        # Simplest possible answer: Yes, always
        return True

