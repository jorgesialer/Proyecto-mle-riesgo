# Sistema de Prediccion de Aprobacion Crediticia

## 1. Definicion del problema

Este proyecto implementa un baseline de Machine Learning para predecir la
variable historica `LoanApproved`: `1` representa una solicitud aprobada y `0`
una solicitud rechazada.

El target describe una decision historica de aprobacion. No representa un
impago observado y, por tanto, las probabilidades producidas por el modelo no
deben interpretarse como probabilidades de default.

## 2. Dataset

El conjunto de datos corresponde a un dataset tabular diseñado para la
clasificacion binaria de aprobacion crediticia con 5,000 registros historicos de
solicitantes, incluyendo variables demograficas, situacion laboral y metricas
financieras clave.

La entrada canonica de entrenamiento es
`data/loan_risk_prediction_dataset.csv`, con 5,000 solicitudes y las siguientes
variables:

| Variable | Tipo | Descripcion |
| :--- | :--- | :--- |
| `Age` | Numerica | Edad del solicitante. |
| `Income` | Numerica | Ingreso anual declarado. |
| `LoanAmount` | Numerica | Monto solicitado. |
| `CreditScore` | Numerica | Puntaje crediticio disponible en la solicitud. |
| `YearsExperience` | Numerica | Anos de experiencia laboral. |
| `Gender` | Categorica | Genero registrado. |
| `Education` | Categorica | Nivel educativo registrado. |
| `City` | Categorica | Ciudad registrada. |
| `EmploymentType` | Categorica | Tipo de empleo registrado. |
| `LoanApproved` | Binaria | Target: aprobada (`1`) o rechazada (`0`). |

Los valores negativos de `Income` y `LoanAmount` se consideran invalidos y se
convierten en valores faltantes. No se asume que sean errores de signo. La
conversion es determinista y ocurre dentro del pipeline antes de la imputacion.

El archivo historico `data/credit_risk_dataset_limpio.csv` se conserva, pero ya
no es una entrada del entrenamiento canonico.

## 3. Pipeline reproducible

El flujo actual es:

```text
CSV crudo
  -> split estratificado train/holdout
  -> validacion cruzada sobre training data
       -> invalidacion de negativos
       -> imputacion numerica por mediana
       -> imputacion categorica por moda
       -> OneHotEncoder(handle_unknown="ignore")
       -> RandomForestClassifier
  -> ajuste del mismo pipeline con todo el training set
  -> evaluacion unica en el holdout final
  -> artifacts/pipeline_rf_baseline.pkl
```

El `ColumnTransformer` y el `RandomForestClassifier` se guardan como un unico
artefacto. Imputadores y encoder se ajustan exclusivamente con training data y,
durante validacion cruzada, exclusivamente con el fold de entrenamiento.
La inferencia exige exactamente las nueve columnas predictoras documentadas y
rechaza columnas faltantes, duplicadas o inesperadas con un error de contrato.

Configuracion del baseline:

- `RandomForestClassifier`
- `n_estimators=100`
- `class_weight="balanced"`
- `random_state=42`
- split estratificado 80/20
- validacion cruzada estratificada de 5 folds sobre el 80% de training

`StandardScaler` no se utiliza porque el baseline es un Random Forest.

## 4. Resultados del baseline saneado

### Validacion cruzada sobre training data

Valores medios y desviacion estandar de 5 folds:

| Metrica | Media | Desviacion estandar |
| :--- | ---: | ---: |
| Precision | 0.965650 | 0.018981 |
| Recall | 0.904495 | 0.025861 |
| F1 | 0.933760 | 0.015653 |
| ROC-AUC | 0.955381 | 0.014626 |
| PR-AUC | 0.918558 | 0.027800 |

### Holdout final

El holdout no se utilizo para seleccion, tuning ni demostraciones:

| Metrica | Resultado |
| :--- | ---: |
| Precision | 0.940092 |
| Recall | 0.886957 |
| F1 | 0.912752 |
| ROC-AUC | 0.944130 |
| PR-AUC | 0.903451 |

En este proyecto, PR-AUC se calcula y reporta mediante Average Precision
(`average_precision_score`).

## 5. Ejecucion

Desde la raiz del repositorio y con el entorno del proyecto activo:

```bash
# Sincronizar el entorno canonico desde pyproject.toml y uv.lock
uv sync

# Ejecutar tests
uv run python -m unittest discover -s tests -v

# Entrenar, validar, evaluar y persistir el pipeline
uv run python -m src.entrenamiento

# Simular inferencia con un perfil crudo
uv run python -m src.prediccion
```

`pyproject.toml` y `uv.lock` son las fuentes canonicas de dependencias. El
proyecto no mantiene un `requirements.txt` paralelo.

`prediccion.py` recibe las variables categoricas originales; no requiere
columnas one-hot ni un scaler externo.

## 6. Estructura relevante

- `data/`: dataset crudo y archivos historicos locales.
- `src/preprocesamiento.py`: contrato de variables y construccion del pipeline.
- `src/entrenamiento.py`: split, CV, evaluacion y persistencia.
- `src/prediccion.py`: inferencia con el pipeline unico.
- `tests/`: pruebas del contrato de preprocesamiento e inferencia.
- `artifacts/pipeline_rf_baseline.pkl`: pipeline evaluado y persistido.
- `notebooks/`: evidencia historica de experimentacion V1.

Los notebooks no son la fuente canonica del pipeline actual y sus resultados
anteriores no deben atribuirse al artefacto saneado.

## 7. Alcance y limitaciones

- El modelo predice aprobaciones historicas, no impago ni capacidad causal de
  repago.
- No se ha realizado todavia seleccion de modelos V2 ni tuning.
- La integracion Gemini existente es un componente legado de V1. No constituye
  explicabilidad basada en evidencia del modelo. Su prompt identifica que el
  resultado proviene de ML y prohibe atribuir causalidad sin evidencia.
- Este baseline no implementa MLflow, nuevas features, XAI, RAG, Qdrant,
  LangGraph ni MCP.
- Los atributos demograficos pueden reproducir sesgos presentes en decisiones
  historicas y requieren una evaluacion de equidad separada.

## 8. Conclusiones

- **Eficacia del Enfoque Hibrido (ML + GenAI):** La combinacion de un modelo clasico de Machine Learning (`RandomForestClassifier`) con un modelo de lenguaje fundacional (`Gemini 2.5 Flash`) demostro ser una arquitectura viable para el sector financiero, integrando la prediccion probabilistica del baseline con una sintesis descriptiva para comites de credito. En fases posteriores V2 se incorporara explicabilidad formal basada en evidencia (XAI) para sustentar los reportes.

- **Robustez y Estabilidad de la Prediccion:** El modelo predictivo alcanzo en el baseline saneado un F1-Score medio de 0.933760 (+/- 0.015653) bajo validacion cruzada estratificada de 5 folds y un F1-Score de 0.912752 en el holdout final (ROC-AUC de 0.944130 y PR-AUC de 0.903451). Estos resultados confirman que el pipeline unificado y fold-safe mitiga el desbalance mediante `class_weight='balanced'` y previene data leakage.

- **Determinacion Contextual Coherente:** Mediante la inferencia simulada en `src/prediccion.py`, el modelo de lenguaje contextualiza los factores financieros respetando las variables de entrada proporcionadas por el pipeline de datos, manteniendo un tono corporativo y reconociendo explicitamente su naturaleza descriptiva sin atribuir causalidad no demostrada.

- **Delimitacion del Alcance Operativo:** Para la version 1.0.0 saneada, el proyecto cumple exitosamente con el objetivo de automatizar de punta a punta el pipeline de datos, el entrenamiento parametrizado, la inferencia y una suite de pruebas unitarias bajo `uv`. Se establece asi una base tecnica solida y reproducible para el desarrollo de las fases V2 (nuevas variables, MLflow, XAI, RAG y orquestacion con LangGraph).
