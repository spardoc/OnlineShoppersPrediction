from smtplib import SMTPException
from alertupload_rest.serializers import UploadAlertSerializer
from rest_framework.response import Response
from rest_framework.decorators import api_view
from django.http import BadHeaderError, JsonResponse
from threading import Thread
from django.core.mail import send_mail
import re

# Thread decorator definition
def start_new_thread(function):
    def decorator(*args, **kwargs):
        t = Thread(target = function, args=args, kwargs=kwargs)
        t.daemon = True
        t.start()
    return decorator

@api_view(['POST'])
def post_alert(request):
    serializer = UploadAlertSerializer(data=request.data)
    if serializer.is_valid():
        serializer.save()
        identify_email_sms(serializer)
    else:
        return JsonResponse({'error': 'Unable to process data!'}, status=400)
    return Response(request.META.get('HTTP_AUTHORIZATION'))

# Identifies if the user provided an email or a mobile number
def identify_email_sms(serializer):
    if(re.search('^[a-z0-9]+[\._]?[a-z0-9]+[@]\w+[.]\w{2,3}$', serializer.data['alert_receiver'])):  
        print("Valid Email")
        send_email(serializer)
    elif re.compile("[+593][0-9]{10}").match(serializer.data['alert_receiver']):
        # 1) Begins with +593
        # 2) Then contains 10 digits 
        print("Valid Mobile Number")
    else:
        print("Invalid Email or Mobile number")

# Sends email
@start_new_thread
def send_email(serializer):
    try:
        send_mail(
            'Weapon Detected!', 
            prepare_alert_message(serializer), 
            'samuelpardo1997@gmail.com',
            [serializer.data['alert_receiver']],
            fail_silently=False,  # Set to False to capture exceptions
        )
        print("Correo enviado exitosamente")
    except Exception as e:
        print("Error al enviar el correo:", e)

# Prepares the alert message
def prepare_alert_message(serializer):
    uuid_with_slashes = split(serializer.data['image'], ".")
    uuid = split(uuid_with_slashes[3], "/")
    url = 'http://127.0.0.1:8000/alert' + uuid[2]
    return 'Weapon Detected! View alert at ' + url

# Splits string into a list
def split(value, key):
    return str(value).split(key)

# views.py
from django.http import JsonResponse
from django.shortcuts import render
import pandas as pd
import joblib

# Cargar el modelo entrenado
modelo_rf = joblib.load('random_forest_model.pkl')

def predecir_compra(request):
    # Recibir los datos como parámetros GET o POST
    try:
        nuevo_sample = {
            'VisitorType_New_Visitor': int(request.GET.get('VisitorType_New_Visitor')),
            'VisitorType_Other': int(request.GET.get('VisitorType_Other')),
            'VisitorType_Returning_Visitor': int(request.GET.get('VisitorType_Returning_Visitor')),
            'Month_Aug': int(request.GET.get('Month_Aug')),
            'Month_Dec': int(request.GET.get('Month_Dec')),
            'Month_Feb': int(request.GET.get('Month_Feb')),
            'Month_Jul': int(request.GET.get('Month_Jul')),
            'Month_June': int(request.GET.get('Month_June')),
            'Month_Mar': int(request.GET.get('Month_Mar')),
            'Month_May': int(request.GET.get('Month_May')),
            'Month_Nov': int(request.GET.get('Month_Nov')),
            'Month_Oct': int(request.GET.get('Month_Oct')),
            'Month_Sep': int(request.GET.get('Month_Sep')),
            'Weekend_False': int(request.GET.get('Weekend_False')),
            'Weekend_True': int(request.GET.get('Weekend_True')),
            'ProductRelated': float(request.GET.get('ProductRelated')),
            'ProductRelated_Duration': float(request.GET.get('ProductRelated_Duration')),
            'BounceRates': float(request.GET.get('BounceRates')),
            'ExitRates': float(request.GET.get('ExitRates')),
            'PageValues': float(request.GET.get('PageValues')),
            'SpecialDay': float(request.GET.get('SpecialDay')),
            'OperatingSystems': int(request.GET.get('OperatingSystems')),
            'Browser': int(request.GET.get('Browser')),
            'Region': int(request.GET.get('Region')),
            'TrafficType': int(request.GET.get('TrafficType')),
        }

        # Convertir el sample a DataFrame
        columnas = ['VisitorType_New_Visitor', 'VisitorType_Other', 'VisitorType_Returning_Visitor',
                    'Month_Aug', 'Month_Dec', 'Month_Feb', 'Month_Jul', 'Month_June', 'Month_Mar',
                    'Month_May', 'Month_Nov', 'Month_Oct', 'Month_Sep', 'Weekend_False', 'Weekend_True',
                    'ProductRelated', 'ProductRelated_Duration', 'BounceRates', 'ExitRates',
                    'PageValues', 'SpecialDay', 'OperatingSystems', 'Browser', 'Region', 'TrafficType']
        df_sample = pd.DataFrame([nuevo_sample], columns=columnas)

        # Realizar la predicción
        prediccion = modelo_rf.predict(df_sample)

        # Interpretar la predicción
        resultado = "Compra" if prediccion[0] == 1 else "No Compra"

        # Retornar la respuesta
        return JsonResponse({'prediccion': resultado})
    
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=400)
        return JsonResponse({'prediccion': resultado})




