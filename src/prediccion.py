"""Inferencia V1 usando el pipeline unico entrenado desde datos crudos."""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import google.generativeai as genai
import pandas as pd
from dotenv import load_dotenv

from .preprocesamiento import validar_columnas_predictoras


load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


class SimuladorInferencia:
    def __init__(self, ruta_pipeline: str):
        print("Cargando pipeline de preprocesamiento y modelo...")
        with Path(ruta_pipeline).open("rb") as archivo:
            self.pipeline = pickle.load(archivo)

        # Integracion Gemini legada de V1; no constituye explicabilidad del modelo.
        self.llm_model = genai.GenerativeModel("gemini-2.5-flash")

    def predecir(self, perfil_cliente: dict):
        df_cliente = pd.DataFrame([perfil_cliente])
        validar_columnas_predictoras(df_cliente.columns)

        prediccion = self.pipeline.predict(df_cliente)[0]
        indice_positivo = list(self.pipeline.classes_).index(1)
        probabilidad = self.pipeline.predict_proba(df_cliente)[0][indice_positivo]

        resultado_str = "APROBADO" if prediccion == 1 else "RECHAZADO"

        print("\n--- Resultado del Sistema ---")
        print(f"Decision predicha: {resultado_str}")
        print(f"Probabilidad de aprobacion: {probabilidad:.2%}")

        print("\nGenerando reporte gerencial con IA...")
        prompt = f"""
        Actúa como un Analista de Riesgos Senior en una institución financiera.
        Un modelo de Machine Learning ha analizado la solicitud y su resultado predictivo es: {resultado_str}.

        Datos financieros y demográficos del perfil:
        - Edad: {perfil_cliente['Age']} años
        - Ingresos Anuales: ${perfil_cliente['Income']}
        - Monto del Préstamo Solicitado: ${perfil_cliente['LoanAmount']}
        - Score Crediticio: {perfil_cliente['CreditScore']} (Escala de referencia: <600 es Alto Riesgo)
        - Experiencia Laboral: {perfil_cliente['YearsExperience']} años

        El resultado anterior proviene de un modelo de Machine Learning. Esta integración legacy no dispone todavía de evidencia de atribución del modelo.

        Tu tarea es redactar un resumen descriptivo, directo y estructurado en un solo párrafo para revisión del Comité de Créditos.

        Reglas estrictas que debes seguir:
        1. Analiza y menciona la relación entre el monto solicitado y los ingresos anuales.
        2. No afirmes que una variable causó o fue determinante para la predicción.
        3. No inventes políticas, umbrales ni evidencia que no aparezcan en los datos proporcionados.
        4. Mantén un tono corporativo, frío y analítico.
        """

        try:
            respuesta = self.llm_model.generate_content(prompt)
            print(f"\n--- Reporte del Analista Virtual ---\n{respuesta.text}")
        except Exception as error:
            print(f"\nError al conectar con Gemini: {error}")


if __name__ == "__main__":
    cliente_nuevo = {
        "Age": 35,
        "Income": 85000,
        "LoanAmount": 20000,
        "CreditScore": 750,
        "YearsExperience": 8,
        "Gender": "Male",
        "Education": "Bachelors",
        "City": "New York",
        "EmploymentType": "Salaried",
    }

    simulador = SimuladorInferencia(
        ruta_pipeline="artifacts/pipeline_rf_baseline.pkl"
    )
    simulador.predecir(cliente_nuevo)
