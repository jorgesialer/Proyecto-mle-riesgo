# Model Card: Baseline de Aprobacion Crediticia

## Identificacion

- **Estado:** baseline V1 saneado, previo a funcionalidades V2.
- **Target:** `LoanApproved`.
- **Estimador:** `RandomForestClassifier`.
- **Parametros:** `n_estimators=100`, `class_weight="balanced"`,
  `random_state=42`.
- **Artefacto:** `artifacts/pipeline_rf_baseline.pkl`.

## Uso previsto

El modelo estima la probabilidad de que una solicitud se clasifique como
historicamente aprobada (`LoanApproved=1`). Puede utilizarse como baseline de
experimentacion y como soporte preliminar para analistas.

No estima probabilidad de impago, no demuestra capacidad de repago y no debe
utilizarse para tomar decisiones autonomas sobre personas.

## Datos y preprocesamiento

La entrada canonica es el dataset crudo
`data/loan_risk_prediction_dataset.csv`, con 5,000 registros.

El split estratificado 80/20 se ejecuta antes de ajustar transformaciones. El
20% se reserva como holdout final. Sobre el 80% de training se ejecuta
validacion cruzada estratificada de 5 folds.

El artefacto persistido incluye:

- conversion determinista de valores negativos de `Income` y `LoanAmount` a
  valores faltantes;
- imputacion numerica por mediana;
- imputacion categorica por moda;
- `OneHotEncoder(handle_unknown="ignore")`, sin eliminar categorias;
- Random Forest.

Los valores negativos se consideran invalidos. No se transforman mediante
valor absoluto porque no existe evidencia de que sean errores de signo.

Todos los pasos aprendidos se ajustan exclusivamente con training data. En CV,
se ajustan nuevamente dentro de cada fold.

## Protocolo de evaluacion

- `train_test_split(test_size=0.2, stratify=y, random_state=42)`.
- `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` sobre training.
- El holdout final no participa en seleccion, tuning ni demostraciones.
- El pipeline se ajusta con todo el training set, se evalua una vez en holdout
  y ese mismo objeto, sin refit posterior, es el que se persiste.
- PR-AUC se reporta mediante Average Precision.

## Resultados de validacion cruzada

Media y desviacion estandar de 5 folds:

| Metrica | Media | Desviacion estandar |
| :--- | ---: | ---: |
| Precision | 0.965650 | 0.018981 |
| Recall | 0.904495 | 0.025861 |
| F1 | 0.933760 | 0.015653 |
| ROC-AUC | 0.955381 | 0.014626 |
| PR-AUC / Average Precision | 0.918558 | 0.027800 |

## Resultados del holdout final

| Metrica | Resultado |
| :--- | ---: |
| Precision | 0.940092 |
| Recall | 0.886957 |
| F1 | 0.912752 |
| ROC-AUC | 0.944130 |
| PR-AUC / Average Precision | 0.903451 |

Estas metricas corresponden exactamente a la configuracion persistida en
`pipeline_rf_baseline.pkl`. No deben mezclarse con resultados historicos de los
notebooks o artefactos V1 anteriores.

## Explicabilidad y GenAI

El baseline no implementa todavia explicabilidad basada en evidencia del
modelo. La integracion Gemini heredada de V1 genera texto, pero no recibe
contribuciones locales ni evidencia causal del Random Forest y no debe
presentarse como XAI. El prompt legacy declara que el resultado proviene de un
modelo de Machine Learning y prohibe afirmar causalidad o factores
determinantes sin evidencia.

## Limitaciones y riesgos

- `LoanApproved` representa una decision historica, no un outcome de default.
- No se ha evaluado generalizacion temporal o entre instituciones.
- No existen identificadores ni fechas para auditar separacion por solicitante
  o por periodo.
- `Gender`, `City` y otros atributos pueden reproducir sesgos historicos.
- `handle_unknown="ignore"` permite inferencia con categorias nuevas, que se
  codifican como un vector de ceros para esa variable y deben monitorearse. El
  encoder conserva todas las categorias conocidas.
- El baseline no incluye nuevas features V2, MLflow, XAI, RAG, Qdrant,
  LangGraph ni MCP.
