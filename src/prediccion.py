import pandas as pd
import pickle
import google.generativeai as genai
import os
from dotenv import load_dotenv

# 1. Cargar variables de entorno ocultas
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

class SimuladorInferencia:
    def __init__(self, ruta_modelo: str, ruta_scaler: str):
        print("Cargando modelo y escalador en memoria...")
        with open(ruta_modelo, 'rb') as f:
            self.modelo = pickle.load(f)
        with open(ruta_scaler, 'rb') as f:
            self.scaler = pickle.load(f)
            
        # Inicializar el modelo de lenguaje de Google
        self.llm_model = genai.GenerativeModel('gemini-2.5-flash')
            
    def predecir(self, perfil_cliente: dict):
        df_cliente = pd.DataFrame([perfil_cliente])
        
        # 1. Escalar numéricos (creamos una copia para no alterar los datos originales del prompt)
        df_escalado = df_cliente.copy()
        columnas_numericas = ['Age', 'Income', 'LoanAmount', 'CreditScore', 'YearsExperience']
        df_escalado[columnas_numericas] = self.scaler.transform(df_cliente[columnas_numericas])
        
        # 2. Realizar predicción matemática
        prediccion = self.modelo.predict(df_escalado)[0]
        probabilidad = self.modelo.predict_proba(df_escalado)[0][1]
        
        resultado_str = "APROBADO" if prediccion == 1 else "RECHAZADO"
        
        print("\n--- Resultado del Sistema ---")
        if prediccion == 1:
            print(f"CRÉDITO APROBADO (Certeza matemática: {probabilidad:.2%})")
        else:
            print(f"CRÉDITO RECHAZADO (Probabilidad de aprobación: {probabilidad:.2%})")

        # 3. Integración con LLM (Explainable AI)
        print("\nGenerando reporte gerencial con IA...")
        prompt = f"""
        Actúa como un Analista de Riesgos Senior en una institución financiera. 
        Nuestro algoritmo de evaluación ha analizado a un solicitante y la decisión final es: {resultado_str}.

        Datos financieros y demográficos del perfil:
        - Edad: {perfil_cliente['Age']} años
        - Ingresos Anuales: ${perfil_cliente['Income']}
        - Monto del Préstamo Solicitado: ${perfil_cliente['LoanAmount']}
        - Score Crediticio: {perfil_cliente['CreditScore']} (Escala de referencia: <600 es Alto Riesgo)
        - Experiencia Laboral: {perfil_cliente['YearsExperience']} años

        Tu tarea es redactar un reporte técnico, directo y estructurado en un solo párrafo para el Comité de Créditos, justificando esta decisión. 

        Reglas estrictas que debes seguir:
        1. Analiza y menciona la relación entre el monto solicitado y los ingresos anuales.
        2. Evalúa si el Score Crediticio es el factor determinante.
        3. Mantén un tono corporativo, frío y analítico.
        4. NO menciones que eres una IA, ni que usaste Machine Learning.
        """
        
        try:
            respuesta = self.llm_model.generate_content(prompt)
            print(f"\n--- Reporte del Analista Virtual ---\n{respuesta.text}")
        except Exception as e:
            print(f"\nError al conectar con Gemini: {e}")

if __name__ == "__main__":
    cliente_nuevo = {
        'Age': 35,
        'Income': 85000,
        'LoanAmount': 20000,
        'CreditScore': 750,
        'YearsExperience': 8,
        'Gender_Male': 1,
        'Education_High School': 0,
        'Education_Masters': 0,
        'Education_PhD': 0,
        'City_Houston': 0,
        'City_New York': 1,
        'City_San Francisco': 0,
        'EmploymentType_Self-Employed': 0,
        'EmploymentType_Unemployed': 0
    }
    
    simulador = SimuladorInferencia(
        ruta_modelo='artifacts/modelo_rf_final.pkl',
        ruta_scaler='artifacts/scaler.pkl'
    )
    simulador.predecir(cliente_nuevo)