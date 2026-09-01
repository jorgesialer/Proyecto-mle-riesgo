# Model Card: HMDA Loan Approval V2

## Identificacion

- **Estado:** champion V2 validado; no es un modelo aprobado para produccion.
- **Target:** `LoanApproved` (`1` aprobada/originada, `0` denegada).
- **Configuracion seleccionada:** `catboost_tuned_02`.
- **Estimador:** `CatBoostClassifier`.
- **Parametros principales:** `iterations=450`, `depth=7`,
  `learning_rate=0.04`, `random_seed=42`.
- **Artefacto local:** `artifacts/pipeline_champion_v2.pkl`.
- **Modelo registrado:** `credit-approval-v2`, version 1.
- **Dataset:** `hmda_2023_loan_approval_v2`.
- **Dataset SHA-256:**
  `57408ac78a0db54d9ed94f805e00a300b473f1232f5fed5650fa5fa3105dc36b`.

## Uso previsto

El modelo estima la probabilidad de que una solicitud HMDA comparable se
clasifique como historicamente aprobada (`LoanApproved=1`). Sirve para
experimentacion, analisis del comportamiento historico y demostracion de un
pipeline MLE reproducible.

No estima default, repago ni capacidad crediticia causal. No debe tomar
decisiones autonomas sobre personas ni utilizarse como politica normativa de
credito.

## Datos y contrato de entrada

El dataset contiene 50,000 aplicaciones HMDA 2023 elegibles. El pipeline recibe
exactamente 16 variables crudas y genera seis features deterministas antes de
imputacion y encoding. Imputadores y `OneHotEncoder(handle_unknown="ignore")`
fueron ajustados exclusivamente con training data.

`applicant_age`, `derived_race`, `derived_ethnicity` y `derived_sex` se
conservan solo para futuras auditorias de fairness. Una whitelist impide que
entren al pipeline, al estimador y a las explicaciones SHAP.

## Seleccion y evaluacion

El split estratificado reserva 20% (10,000 filas) como holdout final. Once
configuraciones de Random Forest, XGBoost y CatBoost se compararon mediante CV
estratificada de cinco folds sobre las 40,000 filas de training. La seleccion
uso F1 medio, con Average Precision y ROC-AUC como criterios secundarios.

### Validacion cruzada del champion

| Metrica | Media | Desviacion estandar |
| :--- | ---: | ---: |
| Precision | 0.883045 | 0.002088 |
| Recall | 0.966567 | 0.001097 |
| F1 | 0.922920 | 0.001618 |
| ROC-AUC | 0.878539 | 0.005286 |
| Average Precision | 0.952849 | 0.002374 |

### Holdout final

| Metrica | Resultado |
| :--- | ---: |
| Precision | 0.881443 |
| Recall | 0.965606 |
| F1 | 0.921607 |
| ROC-AUC | 0.874957 |
| Average Precision | 0.951897 |

Matriz de confusion `[[1196, 1012], [268, 7524]]`, con labels `[0, 1]`.
El holdout no participo en comparacion, tuning ni seleccion.

La diferencia de F1 CV frente a `xgb_baseline_full` es aproximadamente 0.0006,
menor que la variabilidad entre folds. Por ello, CatBoost es el champion del
protocolo acordado, pero no existe evidencia de superioridad contundente sobre
XGBoost.

## Sensibilidad a workflow features

La configuracion de robustez retiro `income`, DTI, CLTV y `property_value`,
ademas de sus features derivadas. Para CatBoost, el F1 CV disminuyo de 0.921375
a 0.890845 (delta -0.030530); Average Precision disminuyo -0.033284 y ROC-AUC
-0.083523. Degradaciones consistentes tambien aparecieron en Random Forest y
XGBoost.

El champion depende materialmente de informacion cuya disponibilidad puede
estar relacionada con la profundidad del underwriting. Esta dependencia debe
monitorearse ante cambios de proceso, institucion y patrones de missingness.

## Explicabilidad

La capa `src/xai_v2.py` utiliza Tree SHAP sobre el CatBoost persistido. La
explicacion global usa 1,000 filas reproducibles del partition de training
(`random_state=42`) y registra mean absolute SHAP, beeswarm, bar plot y rankings
tabulares. Las explicaciones locales devuelven probabilidad de la clase
`LoanApproved=1`, base value y contribuciones positivas/negativas.

El mapping conserva:

- feature de negocio y fuentes crudas;
- features financieras derivadas;
- categoria one-hot y si estaba presente;
- valor observado, valor codificado y contribucion SHAP.

Las enumeraciones principales de proposito, tipo de prestamo, lien, ocupacion y
metodo de construccion usan etiquetas oficiales HMDA, manteniendo el codigo
original. Para el ranking global, las contribuciones transformadas de cada
feature se suman primero dentro de cada fila y luego se calcula el promedio de
su valor absoluto. El detalle de las 45 columnas transformadas permanece
disponible como artifact separado.

Las contribuciones SHAP del CatBoost binario estan en raw margin/log-odds. No
son cambios directos de probabilidad y no demuestran causalidad. El paquete de
evidencia es independiente del LLM; Gemini, RAG y LangGraph no consumen todavia
estas explicaciones.

## Fairness

Los atributos protegidos/proxy audit-only no se usan para predecir ni explicar.
Su exclusion no demuestra ausencia de sesgo: variables no protegidas pueden
actuar como proxies y el target refleja decisiones historicas institucionales.
No se ha implementado aun un fairness dashboard ni metricas por subgrupo.

## Limitaciones y riesgos

- HMDA registra aprobacion historica, no default ni calidad normativa de la
  decision.
- No se evaluo generalizacion temporal fuera de 2023 ni entre instituciones.
- Income, DTI, CLTV y property value presentan missingness dependiente del
  resultado y pueden codificar diferencias del flujo de underwriting.
- Los valores HMDA publicados incorporan modificaciones de privacidad.
- SHAP explica el comportamiento del modelo para sus inputs; no valida la
  correccion de la decision ni proporciona explicaciones causales.
- La explicacion global usa una muestra, no las 50,000 filas.
- La agregacion por feature permite cancelacion entre contribuciones one-hot de
  la misma fila, coherente con la aditividad SHAP. El detalle transformado se
  conserva para diagnosticar esas cancelaciones.
- RAG, Qdrant, LangGraph, MCP, nueva UI y GenAI grounded no estan implementados.

## Evidencia y reproducibilidad

- Benchmark: `artifacts/benchmark_v2_summary.json`.
- XAI local: `artifacts/xai/`.
- Run XAI:
  <https://dagshub.com/jorgesialer/Proyecto-mle-riesgo.mlflow/#/experiments/0/runs/f8e8fc291f8749a9b18634c5530ee054>.
- La run registra metodo, modelo/version, muestra, semilla, top N, checksum y
  todos los artifacts XAI.
