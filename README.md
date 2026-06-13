# Sistema de Evaluación de Riesgo Crediticio

## 1. Definición del Problema y Contexto

En el sector financiero, la evaluación de solicitudes de crédito requiere un equilibrio estricto entre precisión analítica y velocidad de respuesta. Los métodos tradicionales basados únicamente en reglas manuales son susceptibles a cuellos de botella operativos y sesgos cognitivos. 

Este proyecto propone e implementa un sistema automatizado de soporte a la toma de decisiones que utiliza un modelo predictivo de Machine Learning (`RandomForestClassifier`) para evaluar instantáneamente la probabilidad de impago de un cliente. Adicionalmente, el sistema integra capacidades de Inteligencia Artificial Generativa (`Google Gemini API`) como capa de explicabilidad (Explainable AI - XAI), traduciendo el output probabilístico en un reporte gerencial estructurado para la validación final del Comité de Créditos.

## 2. Diccionario de Datos

**Descripción del Dataset:**
El conjunto de datos utilizado (`credit_risk_dataset.csv`) es un dataset tabular diseñado para la clasificación binaria de riesgo financiero. Contiene información histórica de 5,000 solicitantes, incluyendo variables demográficas (edad, género, ciudad), situación laboral (experiencia, tipo de empleo) y métricas financieras clave (ingresos, monto del préstamo, historial de score crediticio). El objetivo principal es predecir la viabilidad de otorgar un crédito mitigando el riesgo de impago.

| Variable | Tipo de Dato | Descripción |
| :--- | :--- | :--- |
| `Age` | Continuo | Edad del solicitante en años. |
| `Income` | Continuo | Ingresos netos anuales del solicitante expresados en USD. |
| `LoanAmount` | Continuo | Monto total del préstamo solicitado en USD. |
| `CreditScore` | Continuo | Puntuación crediticia histórica del cliente (escala estándar, valores < 600 representan alto riesgo). |
| `YearsExperience` | Continuo | Años de experiencia laboral comprobable del solicitante. |
| `Gender` | Binario (0/1) | Variable categórica codificada (One-Hot) sobre el género. |
| `Education` | Binario (0/1) | Variable categórica codificada (One-Hot) sobre el nivel educativo máximo alcanzado. |
| `City` | Binario (0/1) | Variable categórica codificada (One-Hot) sobre la ciudad de residencia. |
| `EmploymentType` | Binario (0/1) | Variable categórica codificada (One-Hot) sobre la situación laboral actual. |
| `LoanApproved` | Binario (0/1) | **Variable Objetivo (Target)**: Representa la aprobación (`1`) o el rechazo (`0`) del crédito. |

## 3. Diagrama de Flujo del Sistema
El siguiente esquema ilustra la arquitectura de la solución, desde la ingesta de datos hasta la emisión del reporte generado por la IA.
``mermaid
graph TD
    A[1. Ingesta de Datos: Solicitud de Crédito] --> B[2. Preprocesamiento: Limpieza y Escalado]
    B --> C[3. Inferencia ML: Modelo Random Forest]
    C --> D{Probabilidad de Impago}
    D -->|Alta Probabilidad| E[Decisión: Rechazado]
    D -->|Baja Probabilidad| F[Decisión: Aprobado]
    E --> G[4. Integración XAI: Prompt + Datos]
    F --> G[4. Integración XAI: Prompt + Datos]
    G --> H[API Google Gemini 2.5 Flash]
    H --> I[5. Output Final: Veredicto + Reporte Gerencial]``

## 4. Tarjeta del Modelo (Model Card)

Para un análisis técnico exhaustivo sobre la arquitectura del modelo, los datos de entrenamiento, consideraciones éticas y limitaciones, por favor consulta el documento adjunto:

**[Ver Model Card Detallado](model-card.md)**

## 5. Resultados y Métricas

El modelo fue evaluado utilizando un conjunto de datos de prueba (test set) que representa el 20% de los datos originales, obteniendo los siguientes resultados estáticos (offline):

* **Accuracy (Exactitud):** 96.40%
* **Precision (Precisión):** 95.33%
* **Recall (Sensibilidad):** 88.70%
* **F1-Score:** 91.89%
* **Validación Cruzada (K-Fold=5):** F1-Score promedio de 93.31%, lo que demuestra que el modelo es estable y no sufre de sobreajuste.

*Nota sobre el alcance:* Para la versión 1.0.0 de este proyecto, la evaluación se limita estrictamente a métricas estáticas (offline). La implementación de telemetría y métricas dinámicas (online) en un entorno de producción real queda fuera del alcance de esta entrega y se considerará para futuras iteraciones.

## 6. Estrategia de Git Utilizada

Para el desarrollo de este proyecto se implementó un flujo de trabajo estructurado para el control de versiones:
* Se mantuvo una rama `main` protegida que contiene únicamente el código estable y funcional.
* El desarrollo iterativo (creación de notebooks, scripts de Python y redacción de documentación) se realizó en una rama paralela llamada `development`.
* La integración de los cambios finales se ejecutó mediante un **Pull Request (PR)** hacia la rama `main`. Finalmente, se generó un *Release v1.0.0* para empaquetar la versión final del sistema.

## 7. Estructura del Repositorio y Ejecución

El proyecto sigue los principios SOLID y de Programación Orientada a Objetos. 

### Organización
* `data/`: Contiene los conjuntos de datos crudos y limpios (`.csv`).
* `notebooks/`: Entornos de experimentación y Análisis Exploratorio de Datos.
* `artifacts/`: Archivos binarios serializados (`.pkl`) del modelo entrenado y el escalador.
* `src/`: Scripts de código fuente para uso en producción.

### Ejecución
Asegúrese de tener el entorno virtual activo e instale las dependencias desde `requirements.txt`. El flujo de ejecución es el siguiente:

```bash
# 1. Ejecutar pipeline de limpieza de datos
python src/preprocesamiento.py

# 2. Entrenar el modelo y generar los artefactos (.pkl)
python src/entrenamiento.py

# 3. Simular la inferencia en producción (Evaluación + LLM)
python src/prediccion.py

