from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
from django.conf import settings
from urllib import parse
import numpy as np

def paginate(request, records):
	page = request.GET.get('page', 1)
	paginator = Paginator(records, settings.RECORD_PER_PAGE)
	try:
		records = paginator.page(page)
	except PageNotAnInteger:
		records = paginator.page(1)
	except EmptyPage:
		records = paginator.page(paginator.num_pages)
	return records




