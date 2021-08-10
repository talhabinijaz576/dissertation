from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.contrib.auth.decorators import login_required
from pprint import pprint

@login_required(login_url="/accounts/login/")
def index(request):
	print("Home |  User is authenticated? ",request.user.is_authenticated)

	if(request.user.is_superuser):
		 return redirect('/staff/')

	elif(request.user.user_type=="PROJECT"):
		return redirect('/administration/')

	elif(request.user.user_type=="USER"):
		return redirect('/portal/')

	else:
		return HttpResponse("You are at home. WORK IN PROGRESS")


