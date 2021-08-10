from django.shortcuts import render
from django.http import HttpResponse, HttpResponseNotFound
from rest_framework.response import Response
from foo_auth.models import User, ProjectConfig
from pprint import pprint
from django.shortcuts import render, redirect
from django.contrib.auth import login, authenticate
from django.contrib.sites.shortcuts import get_current_site
from django.utils.encoding import force_bytes, force_text
from django.utils.http import urlsafe_base64_encode, urlsafe_base64_decode
from django.template.loader import render_to_string
from login_app.tokens import password_reset_token
from django.conf import settings
from django.core.mail import send_mail, EmailMessage
from django.core.validators import validate_email
from django.core.exceptions import ValidationError
from pprint import pprint



def PasswordResetRequestView(request):

    password_reset_request_template = "registration/password_reset.html"
    password_reset_sent_template = "registration/password_reset_sent.html"
    password_reset_email_template = "emails/password_reset_email.html"
    password_username_notfound_template = "registration/password_reset_notfound.html"
    context = {}

    if (request.POST):

        username = request.POST.get("username", "")
        try:
            user = User.objects.get(username=username)
            email = user.email if (user.user_type=="PROJECT") else user.account_owner.email
            mail_subject = 'Reset Password for Jazee Bot Command Center'
            token = urlsafe_base64_encode(force_bytes(user.pk))+"137jzt"+password_reset_token.make_token(user)
            current_site = request.scheme + "://" + request.META["HTTP_HOST"]
            reset_url = current_site + "/login/password_reset/form?token={} ".format(token)
            message = render_to_string(password_reset_email_template, {
                'email': email,
                'user': user,
                'reset_url': reset_url
            })
            send_mail(
                mail_subject, message, from_email = settings.REGISTRATION_EMAIL_USER , recipient_list=[email], auth_user=settings.REGISTRATION_EMAIL_USER ,auth_password=settings.REGISTRATION_EMAIL_PASSWORD
            )
            #email.send()
            context["email"] = email
            context["username"] = username
            response = render(request, template_name=password_reset_sent_template, context=context)

        except:
            context["username"] = username
            response = render(request, template_name=password_username_notfound_template, context=context)

    else:
        response = render(request, template_name=password_reset_request_template, context=context)

    return response



def PasswordResetView(request):
    
    reset_form_template = "registration/password_reset_form.html"
    reset_successful_template = "registration/password_reset_successful.html"
    reset_unsuccessful_template = "registration/activation_unsuccessful.html"
    context = {}

    if(request.POST):

        uidb64 = request.POST.get("uid", "")
        token = request.POST.get("token", "")
        password1 = request.POST.get("password", "p1")
        password2 = request.POST.get("confirm_password", "p2")

        try:
            uid = force_text(urlsafe_base64_decode(uidb64))
            user = User.objects.get(pk=uid)

            if password_reset_token.check_token(user, token) and password1==password2 and len(password1)>5:
                user.set_password(password1)
                user.save()
                response = render(request, template_name = reset_successful_template, context=context)
            else:
                raise Exception("Exception")

        except:
            response = render(request, template_name = reset_unsuccessful_template, context=context)

    else:
        try:

            uidb64, token = request.GET.get("token", "").split("137jzt")
            uid = force_text(urlsafe_base64_decode(uidb64))
            user = User.objects.get(pk=uid)
            if password_reset_token.check_token(user, token):
                context["token"] = password_reset_token.make_token(user)
                context["uuid"] = urlsafe_base64_encode(force_bytes(user.pk))
                response = render(request, template_name = reset_form_template, context=context)
            else:
                raise Exception("Exception")

        except:
            response = redirect("/accounts/login/")

    return response


def ForgotUsernameView(request):

    forgot_username_template = "registration/forgot_username.html"
    forgot_username_email_template = "emails/forgot_username_email.html"
    username_sent_template = "registration/forgot_username_sent.html"
    username_notfound_template = "registration/forgot_username_notfound.html"
    context = {}

    if (request.POST):

        email = request.POST.get("email", "")
        users = User.objects.filter(email=email, user_type="dialer", is_active=True)

        if(users.count() > 0):

            #project_users = users.filter(user_type="PROJECT", is_active=True)
            #if( project_users.exists() ):
            #    users = project_users

            mail_subject = 'Username for Jazee Bot Command Center'
            message = render_to_string(forgot_username_email_template, {'users': users,})
            send_mail(
                mail_subject, message, from_email = settings.REGISTRATION_EMAIL_USER , recipient_list=[email], auth_user=settings.REGISTRATION_EMAIL_USER ,auth_password=settings.REGISTRATION_EMAIL_PASSWORD
            )
            context["email"] = email
            response = render(request, template_name=username_sent_template, context=context)

        else:
            context["email"] = email
            response = render(request, template_name=username_notfound_template, context=context)

    else:
        response = render(request, template_name=forgot_username_template, context=context)

    return response