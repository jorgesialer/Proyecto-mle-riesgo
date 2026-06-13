# Model Card: Clasificador de Riesgo Crediticio

## Detalles del Modelo

* **Desarrollador:** Jorge Sialer
* **Fecha del Modelo:** Mayo 2026
* **Versión:** 1.0.0
* **Tipo de Modelo:** `RandomForestClassifier` (Ensamble de Árboles de Decisión).
* **Parámetros Principales:** `n_estimators=100`, `class_weight='balanced'`, estado pseudoaleatorio fijado (`random_state=42`) para reproducibilidad.
* **Técnica de Entrenamiento:** Aprendizaje Supervisado para clasificación binaria. Las variables numéricas continuas fueron estandarizadas mediante `StandardScaler` basándose en la distribución de la muestra de entrenamiento.

## Uso Previsto (Intended Use)

* **Casos de uso principales:** Asistencia automatizada a analistas de riesgos para la evaluación preliminar de solicitudes de crédito de consumo o personales. El modelo está diseñado para emitir un juicio probabilístico y proporcionar una explicación contextualizada a través de un modelo de lenguaje fundacional (`Gemini 2.5 Flash`).
* **Casos de uso fuera de alcance:** No está destinado a la aprobación o rechazo de créditos de forma 100% autónoma sin la supervisión de un analista humano (Human-in-the-Loop), ni para la evaluación de créditos hipotecarios o corporativos complejos, debido a la limitación de las variables de entrada.

## Factores y Atributos

* **Variables Predictoras Clave:** Se identifican los factores financieros (Ingresos Anuales, Puntuación Crediticia, Monto del Préstamo) como los atributos de mayor peso en la función de decisión del ensamble. 
* **Factores Demográficos:** El modelo evalúa la edad y el género. Es imperativo tener en cuenta la posible existencia de correlaciones históricas que puedan afectar la equidad del modelo (ver sección de Consideraciones Éticas).

## Datos de Entrenamiento y Evaluación

* **Origen de Datos:** Dataset financiero extraído de registros históricos de predicción de riesgo (`credit_risk_dataset.csv`).
* **División del Conjunto:** El dataset original de 5000 registros fue dividido mediante muestreo estratificado (`stratify=y`) para mantener la proporción de la variable objetivo. Se destinó el 80% (4000 registros) para el ajuste (entrenamiento) y el 20% (1000 registros) para la evaluación.
* **Preprocesamiento:** Imputación de valores nulos utilizando medidas de tendencia central (mediana y moda). Codificación de variables nominales categóricas mediante la técnica One-Hot Encoding (`drop_first=True` para mitigar la multicolinealidad).

## Análisis Cuantitativo (Métricas)

El modelo fue evaluado utilizando métricas robustas para conjuntos de datos con clases desbalanceadas:

* **Accuracy Global:** 96.40%
* **Precision (Clase Positiva):** 95.33% - Indica una baja tasa de Falsos Positivos, minimizando el riesgo de aprobar créditos a perfiles morosos.
* **Recall (Clase Positiva):** 88.70% - Indica la proporción de clientes solventes que el modelo identificó correctamente. 
* **F1-Score:** 91.89% - Media armónica que confirma el equilibrio general de la clasificación.
* **Validación Cruzada:** El F1-Score promedio en un `K-Fold` de 5 particiones confirma la estabilidad paramétrica, descartando sobreajuste (overfitting) en la muestra particular de entrenamiento.

## Evaluación Cualitativa (Explicabilidad - XAI)

Para validar la integración del modelo de Machine Learning con el modelo de lenguaje generativo (Gemini 2.5 Flash), se realizó una prueba de inferencia simulando un perfil de cliente ficticio con las siguientes características clave: Edad de 35 años, Ingresos anuales de $85,000, Score Crediticio de 750 y 8 años de experiencia laboral.

**Resultado del Modelo Predictivo:** * CRÉDITO APROBADO (Certeza matemática: 88.00%)

**Reporte Generado por la IA (Output real del script `prediccion.py`):**
> "El análisis del perfil del solicitante, con base en nuestros parámetros de evaluación, recomienda la aprobación del préstamo por $20,000. La relación entre el monto solicitado y los ingresos anuales de $85,000 es altamente favorable, representando una carga de deuda manejable y una sólida capacidad de repago. Si bien múltiples factores contribuyen a la decisión, el Score Crediticio de 750, que supera considerablemente el umbral de alto riesgo (<600), emerge como un indicador determinante de la excelente solvencia y comportamiento crediticio del solicitante. Este puntaje, combinado con una edad de 35 años y 8 años de experiencia laboral, sugiere estabilidad financiera y un bajo perfil de riesgo de incumplimiento, justificando la decisión de aprobación."

**Conclusión de la prueba:** El sistema logra exitosamente el objetivo de Explainable AI. La API traduce correctamente el output probabilístico del Random Forest en una justificación de negocio coherente, utilizando las variables de entrada reales sin alucinaciones y manteniendo el tono corporativo requerido para un Comité de Créditos.

## Consideraciones Éticas y Limitaciones

* **Equidad Algorítmica (Fairness):** Los datos históricos de instituciones financieras suelen contener sesgos latentes respecto al género o la ubicación geográfica. Aunque el algoritmo alcanzó altas métricas globales, se recomienda realizar auditorías de paridad demográfica en el futuro para asegurar que las tasas de falsos positivos y negativos no afecten desproporcionadamente a grupos específicos.
* **Explicabilidad:** Dado que un Random Forest opera parcialmente como una "caja negra" debido a su naturaleza de ensamble complejo, este sistema mitiga la falta de interpretabilidad directa conectando el vector de características de entrada y el output predictivo a una API de Generative AI, obligando al sistema a justificar lógicamente la decisión financiera en lenguaje natural.
* **Limitación Temporal:** El modelo no incorpora variables macroeconómicas dinámicas (ej. tasas de inflación actuales, tasas de interés del banco central), asumiendo que las condiciones de mercado del momento en que se recolectaron los datos se mantienen constantes.