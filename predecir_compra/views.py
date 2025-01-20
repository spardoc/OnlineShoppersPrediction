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




