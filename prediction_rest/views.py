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
            # otros datos omitidos por brevedad
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

        # Guardar predicción y explicación en la lista
        predicciones_lista.append({
            'prediction': result,
            'explanation': explanation
        })

        # Redirigir a la página de resultados
        return redirect(reverse('results'))

    except Exception as e:
        print(f"Error: {str(e)}")
        return Response({'error': str(e)}, status=400)


