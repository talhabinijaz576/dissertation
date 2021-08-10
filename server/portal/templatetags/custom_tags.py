from django import template
from django.conf import settings

register = template.Library()

@register.filter
def multiply(value, arg):
    try: 
        return float(value)* arg
    except: 
        return ""

@register.filter
def subtract(value, arg):
    try: 
        return float(value) -  arg
    except: 
        return ""


@register.filter
def ConvertToInt(value):
    #print("Value: ", value)
    try:
        return int(value)
    except:
        return ""
    return value


@register.filter
def DecideIsShown(value, arg):
    # value: records ,  arg: i
    #print("INFO: ", value.number, arg, value.paginator.num_pages)
    num_pages_to_show = 5
    show =  ( value.number <= num_pages_to_show//2 and arg <= num_pages_to_show ) or \
            ( value.number >= ( value.paginator.num_pages- (num_pages_to_show//2 - 1) ) and arg >= (value.paginator.num_pages - num_pages_to_show + 1) ) or \
            ( arg <= (value.number + num_pages_to_show//2) and arg >= (value.number - num_pages_to_show//2) ) 
    
    return show


@register.filter
def ProcessDateforHTML(value):

    if(value==None):
        return ""
        
    original_value = value
    try:
        if(type(value)!=str):
            value = value.strftime(settings.TABLE_DATE_FORMAT)
        
        elif(len(value)<5):
            return value

        value = value.strip()
        pieces = value.split(" ")
        value = pieces[0][:3] +" "+ " ".join(pieces[1:])
        value = value[:-3] + " " + value[-3].lower() + ".m."
        value = value.replace(" 0", " ").replace(". 0", ", ")
        value = value.replace("m.", "m")
        value = value.replace("  ", " ")
    except:
        value = original_value
    
    return value 