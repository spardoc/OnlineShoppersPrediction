from smtplib import SMTPException
from prediction_rest.serializers import UploadAlertSerializer
from rest_framework.response import Response
from rest_framework.decorators import api_view
from django.http import BadHeaderError, JsonResponse
from threading import Thread
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
import pandas as pd
import joblib
import re

# Thread decorator definition
def start_new_thread(function):
    def decorator(*args, **kwargs):
        t = Thread(target = function, args=args, kwargs=kwargs)
        t.daemon = True
        t.start()
    return decorator

# Splits string into a list
def split(value, key):
    return str(value).split(key)

# Cargar el modelo entrenado
modelo_rf = joblib.load('random_forest_model.pkl')

@csrf_exempt
@api_view(['POST'])
def predecir_compra(request):
    try:
        # Recibir los datos como JSON
        body = request.data

        # Mapear los datos al formato esperado
        nuevo_sample = {
            'VisitorType_New_Visitor': int(body.get('VisitorType_New_Visitor')),
            'VisitorType_Other': int(body.get('VisitorType_Other')),
            'VisitorType_Returning_Visitor': int(body.get('VisitorType_Returning_Visitor')),
            'Month_Aug': int(body.get('Month_Aug')),
            'Month_Dec': int(body.get('Month_Dec')),
            'Month_Feb': int(body.get('Month_Feb')),
            'Month_Jul': int(body.get('Month_Jul')),
            'Month_June': int(body.get('Month_June')),
            'Month_Mar': int(body.get('Month_Mar')),
            'Month_May': int(body.get('Month_May')),
            'Month_Nov': int(body.get('Month_Nov')),
            'Month_Oct': int(body.get('Month_Oct')),
            'Month_Sep': int(body.get('Month_Sep')),
            'Weekend_False': int(body.get('Weekend_False')),
            'Weekend_True': int(body.get('Weekend_True')),
            'ProductRelated': float(body.get('ProductRelated')),
            'ProductRelated_Duration': float(body.get('ProductRelated_Duration')),
            'BounceRates': float(body.get('BounceRates')),
            'ExitRates': float(body.get('ExitRates')),
            'PageValues': float(body.get('PageValues')),
            'SpecialDay': float(body.get('SpecialDay')),
            'OperatingSystems': int(body.get('OperatingSystems')),
            'Browser': int(body.get('Browser')),
            'Region': int(body.get('Region')),
            'TrafficType': int(body.get('TrafficType')),
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
        return Response({'prediccion': resultado})
    
    except Exception as e:
        return Response({'error': str(e)}, status=400)
