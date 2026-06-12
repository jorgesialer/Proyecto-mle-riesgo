# Sistema de Evaluación de Riesgo Crediticio

## 1. Definición del Problema y Contexto

En el sector financiero, la evaluación de solicitudes de crédito requiere un equilibrio estricto entre precisión analítica y velocidad de respuesta. Los métodos tradicionales basados únicamente en reglas manuales son susceptibles a cuellos de botella operativos y sesgos cognitivos. 

Este proyecto propone e implementa un sistema automatizado de soporte a la toma de decisiones que utiliza un modelo predictivo de Machine Learning (`RandomForestClassifier`) para evaluar instantáneamente la probabilidad de impago de un cliente. Adicionalmente, el sistema integra capacidades de Inteligencia Artificial Generativa (`Google Gemini API`) como capa de explicabilidad (Explainable AI - XAI), traduciendo el output probabilístico en un reporte gerencial estructurado para la validación final del Comité de Créditos.

## 2. Diccionario de Datos

Los datos provienen de un conjunto histórico de perfiles financieros preprocesados. A continuación, se detalla la estructura del conjunto de datos limpio utilizado para el entrenamiento y la inferencia:

| Variable | Tipo de Dato | Descripción |
| :--- | :--- | :--- |
| `Age` | Continuo | Edad del solicitante en años. |
| `Income` | Continuo | Ingresos netos anuales del solicitante expresados en USD. |
| `LoanAmount` | Continuo | Monto total del préstamo solicitado en USD. |
| `CreditScore` | Continuo | Puntuación crediticia histórica del cliente (escala estándar, valores < 600 representan alto riesgo). |
| `YearsExperience` | Continuo | Años de experiencia laboral comprobable del solicitante. |
| `Gender_*` | Binario (0/1) | Variable categórica codificada (One-Hot) sobre el género. |
| `Education_*` | Binario (0/1) | Variable categórica codificada (One-Hot) sobre el nivel educativo máximo alcanzado. |
| `City_*` | Binario (0/1) | Variable categórica codificada (One-Hot) sobre la ciudad de residencia. |
| `EmploymentType_*` | Binario (0/1) | Variable categórica codificada (One-Hot) sobre la situación laboral actual. |
| `LoanApproved` | Binario (0/1) | **Variable Objetivo (Target)**: Representa la aprobación (`1`) o el rechazo (`0`) del crédito. |

## 3. Diagrama de Flujo del Sistema
El siguiente esquema ilustra la arquitectura de la solución, desde la ingesta de datos hasta la emisión del reporte generado por la IA.
graph TD
    A[1. Ingesta de Datos: Solicitud de Crédito] --> B[2. Preprocesamiento: Limpieza y Escalado]
    B --> C[3. Inferencia ML: Modelo Random Forest]
    C --> D{Probabilidad de Impago}
    D -->|Alta Probabilidad| E[Decisión: Rechazado]
    D -->|Baja Probabilidad| F[Decisión: Aprobado]
    E --> G[4. Integración XAI: Prompt + Datos]
    F --> G[4. Integración XAI: Prompt + Datos]
    G --> H[API Google Gemini 2.5 Flash]
    H --> I[5. Output Final: Veredicto + Reporte Gerencial]

## 4. Tarjeta del Modelo (Model Card)

Para un análisis técnico exhaustivo sobre la arquitectura del modelo, los datos de entrenamiento, consideraciones éticas y limitaciones, por favor consulta el documento adjunto:

**[Ver Model Card Detallado](model-card.md)**

## 5. Métricas de Evaluación

### 5.1 Pruebas Offline (Evaluación Estática)
Las métricas presentadas a continuación fueron calculadas utilizando Validación Cruzada (`K-Fold CV`, $K=5$) y un conjunto de prueba aislado (20% del total) para garantizar la capacidad de generalización del modelo:

* **Accuracy (Exactitud):** 0.9640
* **Precision (Precisión):** 0.9533
* **Recall (Sensibilidad):** 0.8870
* **F1-Score:** 0.9189

### 5.2 Pruebas Online (Propuesta de Telemetría en Producción)
Para el despliegue del modelo en un entorno de producción real, se propone implementar un esquema de despliegue progresivo (Canary Release o A/B Testing) monitoreando las siguientes métricas de negocio y de sistema:

1. **Tasa de Morosidad a 90 días (NPL):** Métrica fundamental de negocio para evaluar si la cartera aprobada por el modelo presenta un índice de incumplimiento menor al umbral histórico del banco.
2. **Latencia de Inferencia:** Tiempo total de procesamiento desde el *request* HTTP hasta la entrega del *response* combinado (Predicción ML + Reporte LLM). El SLA objetivo para el componente de Machine Learning debe ser `< 100ms`.
3. **Data Drift:** Monitoreo estadístico continuo (ej. mediante *Population Stability Index*) de las variables demográficas e ingresos para alertar sobre la necesidad de reentrenar el modelo.

## 6. Estructura del Repositorio y Ejecución

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