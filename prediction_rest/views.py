from io import BytesIO
from smtplib import SMTPException

from matplotlib import pyplot as plt
import numpy as np
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
from pyht import Client
from dotenv import load_dotenv
from pyht.client import TTSOptions
import os
import base64
from openai import OpenAI
import openai
from google.cloud import texttospeech
from rest_framework import status





load_dotenv()

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

model_rf = joblib.load('random_forest_model.pkl')

def generar_explicacion(sample_df, prediccion):
    prompt = (
        f"Eres un experto en datos y análisis y estas trabajando con el dataset Online Shoppers Purchasing Intention Dataset." \
        f"Dados los siguientes datos: {sample_df.to_dict(orient='records')[0]}, " \
        f"y la predicción: {'Compra' if prediccion[0] == 1 else 'No Compra'}, "
        "explica de forma clara por qué se tomó esta decisión."
    )
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        explicacion = response.choices[0].message.content.strip()
        return explicacion
    except Exception as e:
        return f"No se pudo generar una explicación: {str(e)}"

def generar_audio(explicacion):
    credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not credentials_path:
        raise ValueError("La variable GOOGLE_APPLICATION_CREDENTIALS no está configurada en el .env")

    client = texttospeech.TextToSpeechClient()

    synthesis_input = texttospeech.SynthesisInput(text=explicacion)

    voice = texttospeech.VoiceSelectionParams(
        language_code="es-ES",
        ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
    )

    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )

    response = client.synthesize_speech(
        input=synthesis_input,
        voice=voice,
        audio_config=audio_config
    )

    audio_base64 = base64.b64encode(response.audio_content).decode("utf-8")

    return audio_base64

# Lista para almacenar las predicciones
predicciones_lista = []

from django.shortcuts import render, redirect
from django.urls import reverse

@csrf_exempt
@api_view(['POST'])
def predecir_compra(request):
    try:
        body = request.data

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

        columnas = ['VisitorType_New_Visitor', 'VisitorType_Other', 'VisitorType_Returning_Visitor',
                    'Month_Aug', 'Month_Dec', 'Month_Feb', 'Month_Jul', 'Month_June', 'Month_Mar',
                    'Month_May', 'Month_Nov', 'Month_Oct', 'Month_Sep', 'Weekend_False', 'Weekend_True',
                    'ProductRelated', 'ProductRelated_Duration', 'BounceRates', 'ExitRates',
                    'PageValues', 'SpecialDay', 'OperatingSystems', 'Browser', 'Region', 'TrafficType']
        df_sample = pd.DataFrame([nuevo_sample], columns=columnas)

        prediction = model_rf.predict(df_sample)
        result = "Compra" if prediction[0] == 1 else "No Compra"
        explanation = generar_explicacion(df_sample, prediction)

        # Generar gráficos y convertirlos a base64
        grafico_importancia = generar_grafico_importancia(model_rf)
        grafico_probabilidades = generar_grafico_probabilidades(model_rf, list(nuevo_sample.values()))

        grafico_importancia_base64 = convertir_imagen_a_base64(grafico_importancia)
        grafico_probabilidades_base64 = convertir_imagen_a_base64(grafico_probabilidades)

        # Explicación (asegúrate de tener la función de explicación implementada)
        explanation = generar_explicacion(df_sample, prediction)
        audio_base64 = generar_audio(explanation)

        # Guardar predicción y explicación en la lista
        predicciones_lista.append({
            'prediction': result,
            'explanation': explanation,
            'audio': audio_base64,
            'grafico_importancia': grafico_importancia_base64,
            'grafico_probabilidades': grafico_probabilidades_base64
        })

        # Redirigir a la página de resultados
        return redirect(reverse('results'))

    except Exception as e:
        print(f"Error: {str(e)}")
        return Response({'error': str(e)}, status=400)

# Función para convertir una imagen generada en memoria a base64
def convertir_imagen_a_base64(fig):
    buffered = BytesIO()
    fig.savefig(buffered, format="png")
    buffered.seek(0)
    imagen_base64 = base64.b64encode(buffered.read()).decode('utf-8')
    plt.close(fig)  # Cierra la figura para liberar memoria y evitar conflictos
    return imagen_base64


def generar_grafico_importancia(modelo_rf):
    importancia_caracteristicas = modelo_rf.feature_importances_
    columnas = list(modelo_rf.feature_names_in_)
    indices_importancia = np.argsort(importancia_caracteristicas)

    # Crear una nueva figura y eje
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(np.array(columnas)[indices_importancia], importancia_caracteristicas[indices_importancia])
    ax.set_xlabel('Importancia')
    ax.set_title('Importancia de las Características para la Predicción')
    fig.tight_layout()
    
    return fig


# Función para generar el gráfico de probabilidades y convertirlo a base64
def generar_grafico_probabilidades(modelo_rf, sample):
    probabilidades = modelo_rf.predict_proba([sample])[0]

    # Crear una nueva figura y eje
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(['No Compra', 'Compra'], probabilidades, color=['red', 'green'])
    ax.set_ylabel('Probabilidad')
    ax.set_title('Probabilidad de Compra para el Nuevo Cliente')
    fig.tight_layout()
    
    return fig
