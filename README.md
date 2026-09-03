# Sistema de Prediccion de Aprobacion Crediticia

## 1. Definicion del problema

Este proyecto implementa un baseline de Machine Learning para predecir la
variable historica `LoanApproved`: `1` representa una solicitud aprobada y `0`
una solicitud rechazada.

El target describe una decision historica de aprobacion. No representa un
impago observado y, por tanto, las probabilidades producidas por el modelo no
deben interpretarse como probabilidades de default.

## 2. Dataset V1 (baseline historico)

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

V1 se conserva como baseline historico independiente. No se enriquece con
columnas sinteticas ni se mezcla con la fuente de V2.

### Dataset V2 aprobado

V2 utiliza datos reales del **HMDA 2023 One Year National Loan-Level Dataset**
oficial de CFPB/FFIEC (fecha de congelamiento: 19 de mayo de 2025). El target
conceptual sigue siendo `LoanApproved`: acciones originadas o aprobadas no
aceptadas se mapean a `1`, y solicitudes denegadas a `0`.

El constructor consume mediante streaming gzip un export oficial filtrado,
sin guardar el LAR nacional completo. Aplica los filtros de poblacion,
reservoir sampling uniforme con semilla 42 y genera:

- `data/hmda_2023_loan_approval_v2.csv`: 50,000 filas y 21 columnas.
- `data/hmda_2023_loan_approval_v2.metadata.json`: fuente, filtros, roles,
  distribucion del target y missingness global/por clase.
- `docs/HMDA_V2_DATA_DICTIONARY.md`: definiciones, missing codes y riesgos de
  leakage.

Construccion reproducible desde la raiz:

```bash
uv run python -m src.construir_dataset_v2
```

Las 16 variables crudas de ML estan separadas de cuatro atributos
`audit_only` (`applicant_age`, `derived_race`, `derived_ethnicity`,
`derived_sex`). El contrato de codigo impide incluir estos ultimos como
predictores. Tras auditar el sample se retiraron `negative_amortization`
(constante), `introductory_rate_period` (93.214% de nulos estructurales) y
`other_nonamortizing_features` (0.088% positivos). `total_units` se conserva
porque existen 868 registros multiunidad distribuidos en cuatro categorias.

### Feature engineering y pipeline V2

`src/preprocesamiento_v2.py` implementa feature engineering determinista para:

- Loan-to-Income y PropertyValue-to-Income, respetando que income HMDA esta en
  miles de USD.
- Loan-to-Property-Value.
- plazo del prestamo en anos.
- normalizacion de DTI en siete bandas interpretables, sin puntos medios
  inventados.
- conteo de los flags contractuales interest-only y balloon payment.

El flujo aprobado para la siguiente fase es:

```text
Dataset V2 crudo
  -> separar audit-only y LoanApproved mediante whitelist
  -> split estratificado 80/20
  -> CV exclusivamente sobre el 80% de training
       -> feature engineering determinista
       -> imputacion numerica por mediana
       -> imputacion categorica por moda
       -> OneHotEncoder(handle_unknown="ignore")
       -> estimador inyectado por el experimento
  -> evaluacion unica del champion en el holdout final
```

El constructor del pipeline exige inyectar explicitamente un estimador. La
seleccion del champion se realiza por separado mediante el benchmark descrito
a continuacion, sin acoplar el preprocesamiento a una familia de modelos.

### Benchmarking V2 y MLflow

La fase de benchmarking compara exclusivamente:

- `RandomForestClassifier` como baseline;
- `XGBClassifier`;
- `CatBoostClassifier`.

El protocolo reserva un holdout estratificado de 20%. El 80% restante se usa
para CV estratificada de cinco folds, robustness checks, tuning acotado y
seleccion por F1 medio; Average Precision y ROC-AUC actuan como desempates. El
holdout se evalua una sola vez despues de seleccionar el champion.

La comparacion de robustez enfrenta el esquema completo con una configuracion
que excluye income, DTI, CLTV, property value y todas las features derivadas
que reutilizan esa informacion. Accuracy no se usa para seleccionar modelos.

Ejecucion:

```bash
uv sync
uv run python -m src.entrenamiento_v2
```

Backend local predeterminado:

```bash
$env:MLFLOW_BACKEND = "local"
uv run python -m src.entrenamiento_v2
uv run mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000
```

La interfaz queda disponible en `http://127.0.0.1:5000`. Los metadatos se
guardan en `mlflow.db` y los artefactos en `mlartifacts/`; ambos se excluyen de
Git.

Para DagsHub deben configurarse de forma segura, sin escribirlas en archivos
versionados:

```bash
$env:MLFLOW_BACKEND = "dagshub"
$env:MLFLOW_TRACKING_URI = "https://dagshub.com/<usuario>/<repositorio>.mlflow"
$env:MLFLOW_TRACKING_USERNAME = "<usuario>"
$env:MLFLOW_TRACKING_PASSWORD = "<token>"
uv run python -m src.entrenamiento_v2
```

La URL de `MLFLOW_TRACKING_URI` abre también la tabla remota de experimentos.
Cada run usa un nombre descriptivo y tags de familia, configuración, etapa,
versión del dataset y estado de selección.

Las metricas V1 y las futuras metricas V2 **no son directamente comparables**:
usan fuentes, poblaciones, variables y procesos generadores de decisiones
distintos.

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

## 5. Resultados del benchmark V2 (HMDA 2023)

El benchmark V2 evalua 11 configuraciones candidatas (Random Forest, XGBoost y
CatBoost) bajo validacion cruzada estratificada de 5 folds sobre el 80% de
training (40,000 registros). El 20% restante (10,000 registros) se reservo de
forma aislada como holdout final.

### Champion seleccionado

El modelo champion seleccionado exclusivamente mediante CV sobre training data
es:

- **Estimador:** `catboost_tuned_02`
- **Familia:** CatBoost
- **Configuracion de features:** `full` (16 predictores crudos + features derivadas)
- **Hiperparametros clave:**
  - `iterations`: 450
  - `depth`: 7
  - `learning_rate`: 0.04

### Metricas de validacion cruzada (CV 5 folds sobre training)

Valores medios y desviacion estandar sobre 40,000 registros de entrenamiento:

| Metrica | Media | Desviacion estandar |
| :--- | ---: | ---: |
| Precision | 0.883045 | 0.002088 |
| Recall | 0.966567 | 0.001097 |
| F1-Score | 0.922920 | 0.001618 |
| ROC-AUC | 0.878539 | 0.005286 |
| Average Precision (PR-AUC) | 0.952849 | 0.002374 |

### Evaluacion final en Holdout

Evaluacion unica del champion ajustado sobre el 100% de training y evaluado en
los 10,000 registros aislados de holdout:

| Metrica | Resultado |
| :--- | ---: |
| Precision | 0.881443 |
| Recall | 0.965606 |
| F1-Score | 0.921607 |
| ROC-AUC | 0.874957 |
| Average Precision (PR-AUC) | 0.951897 |

Matriz de confusion en holdout:

| | Predicho Rechazo (`0`) | Predicho Aprobado (`1`) |
| :--- | ---: | ---: |
| **Real Rechazo (`0`)** | 1,196 (TN) | 1,012 (FP) |
| **Real Aprobado (`1`)** | 268 (FN) | 7,524 (TP) |

### Comparacion con segundo lugar

`catboost_tuned_02` obtuvo el primer lugar del benchmark superando a
`xgb_baseline_full` (CV F1: 0.922287 ± 0.001925) por un margen estrecho de
$\Delta\text{F1} \approx 0.0006$. Dado que esta diferencia es menor que la
variabilidad observada entre folds ($\text{std} \approx 0.0016$), debe
interpretarse como una ventaja marginal y no como una superioridad
contundente entre familias de gradient boosting.

### Tracking y Model Registry en DagsHub

El benchmark ejecuto 12 runs canonicos (11 runs de exploracion/CV + 1 run final
de champion) registrados con parametros, metricas por fold y artefactos en el
servidor remoto de MLflow en DagsHub:

- **Experimento:** `credit-approval-v2-benchmark`
- **Modelo registrado:** `credit-approval-v2` (version 1)
- **Evidencia en DagsHub:**
  - [MLflow Experiment UI](https://dagshub.com/jorgesialer/Proyecto-mle-riesgo.mlflow/#/experiments/0)
  - [MLflow Model Registry](https://dagshub.com/jorgesialer/Proyecto-mle-riesgo.mlflow/#/models/credit-approval-v2)

> **Nota sobre comparabilidad:** Los resultados de V1 y V2 **no son directamente
> comparables**. V1 corresponde a un dataset sintetico/educativo de 5,000 filas
> con 9 predictores, mientras que V2 se basa en 50,000 solicitudes hipotecarias
> reales de HMDA 2023 con 16 predictores crudos y transformaciones financieras
> especificas.

### Explicabilidad XAI con SHAP

`src/xai_v2.py` carga el pipeline champion sin reentrenarlo y aplica Tree SHAP
al estimador CatBoost sobre una muestra reproducible de 1,000 filas del
partition de training (`random_state=42`). SHAP se utiliza para describir como
las features contribuyen al margen del modelo, no para afirmar causalidad,
capacidad de repago ni una politica normativa de credito.

La capa produce un ranking global agrupado por feature de negocio. Para evitar
sesgo por cardinalidad, primero suma por fila las contribuciones de todas las
columnas one-hot de una feature y despues calcula
`mean(abs(grouped_shap_per_row))`. Se conserva tambien el detalle de las 45
columnas transformadas. Las principales enumeraciones HMDA se muestran con
etiquetas oficiales legibles y conservan su `raw_code`; codigos no reconocidos
se marcan `unknown`. Los cuatro atributos `audit_only` nunca alcanzan el
estimador ni los resultados SHAP.

Artifacts reproducibles:

- `artifacts/xai/global_feature_importance.csv` y `.json`;
- `artifacts/xai/global_feature_importance_transformed.csv` y `.json`;
- `artifacts/xai/shap_summary.png`;
- `artifacts/xai/shap_bar.png`;
- `artifacts/xai/local_example.json`;
- `artifacts/xai/metadata.json` y `run_metadata.json`.

Ejecucion:

```bash
uv run python -m src.xai_v2 --sample-size 1000 --top-n 10
```

La run corregida publicada es
[`xai_catboost_champion_v2`](https://dagshub.com/jorgesialer/Proyecto-mle-riesgo.mlflow/#/experiments/0/runs/f8e8fc291f8749a9b18634c5530ee054).
La run XAI original se conserva como evidencia historica y no se sobrescribe.

### RAG de politica crediticia con Qdrant

`src/rag_v2.py` implementa retrieval semantico sin LLM sobre un corpus pequeno
y exclusivamente oficial. El snapshot incluye el chart HMDA 2023 del CFPB y
cuatro secciones acotadas de la Fannie Mae Selling Guide: informacion general
de ingresos, DTI, LTV y CLTV. Cada documento conserva URL, institucion, fecha o
version, fecha de descarga, nombre local y SHA-256 en
`data/rag_sources/corpus_metadata.json`.

Fuentes del snapshot:

- [CFPB: Reportable HMDA Data 2023 reference chart](https://files.consumerfinance.gov/f/documents/cfpb_reportable-hmda-data_regulatory-and-reporting-overview-reference-chart_2023-02.pdf);
- [Fannie Mae B3-3.1-01: General Income Information](https://selling-guide.fanniemae.com/sel/b3-3.1-01/general-income-information);
- [Fannie Mae B3-6-02: Debt-to-Income Ratios](https://selling-guide.fanniemae.com/sel/b3-6-02/debt-income-ratios);
- [Fannie Mae B2-1.2-01: Loan-to-Value Ratios](https://selling-guide.fanniemae.com/sel/b2-1.2-01/loan-value-ltv-ratios);
- [Fannie Mae B2-1.2-02: Combined Loan-to-Value Ratios](https://selling-guide.fanniemae.com/sel/b2-1.2-02/combined-loan-value-cltv-ratios).

La extraccion usa `pypdf` para PDF y `BeautifulSoup` para HTML. El texto se
divide de forma determinista por pagina/seccion en chunks de hasta 1,000
caracteres con 150 caracteres de solapamiento. El corpus corregido produce 139
chunks a partir de 65 secciones. Los embeddings locales normalizados usan
`sentence-transformers/all-MiniLM-L6-v2` (384 dimensiones) y se almacenan en
la coleccion `credit_policy_v2` mediante Qdrant embedded, con distancia coseno.
El almacenamiento runtime versionado bajo `data/qdrant/` no se versiona en Git;
se reconstruye desde los snapshots y el manifest.

La evaluacion manual `data/rag_eval_queries.json` conserva las 12 queries
dirigidas y agrega cuatro casos: uno ambiguo LTV/CLTV, uno multi-source, una
parafrasis menos literal y uno fuera de dominio. Las 12 dirigidas mantuvieron
source hit@5, source recall@5, concept hit@5 y MRR@5 de 1.0. Los tres casos
adicionales in-domain tambien obtuvieron 1.0 en esas metricas; el caso
out-of-domain devolvio resultados, sin rechazo, con top score 0.363723. El
menor top score in-domain fue 0.534716. Esta separacion es solo una observacion
inicial: no se implementa un threshold con una unica query OOD y se requiere
un conjunto de calibracion mas amplio antes de definir abstencion.

Estos resultados no prueban cobertura normativa general ni calidad de una
respuesta generada. `construir_queries_desde_evidence` traduce factores XAI a
consultas mediante templates deterministas, omitiendo estructuralmente
columnas `audit_only`.

La ejecucion corregida se registro como
[`rag_qdrant_retrieval_v2`](https://dagshub.com/jorgesialer/Proyecto-mle-riesgo.mlflow/#/experiments/0/runs/1e98c47241ff449bb5bcb41cb79caf9c)
sin repetir el benchmark ML. La run
[`rag_qdrant_retrieval_v1`](https://dagshub.com/jorgesialer/Proyecto-mle-riesgo.mlflow/#/experiments/0/runs/f57dba116c4544bcbb87bb1aa1af378c)
se conserva como evidencia historica.

Artifacts reproducibles:

- `artifacts/rag/chunks.jsonl`;
- `artifacts/rag/index_metadata.json`;
- `artifacts/rag/retrieval_evaluation.json`;
- `artifacts/rag/example_retrieval.json`;
- `artifacts/rag/run_metadata.json` tras registrar MLflow/DagsHub.

Ejecucion sin generar respuestas ni invocar Gemini:

```bash
uv run python -m src.rag_v2 --skip-mlflow
```

### Orquestacion determinista y generacion grounded

`src/langgraph_v2.py` compila un `StateGraph` lineal, sin agente ReAct ni
seleccion autonoma de herramientas:

```text
validate_input -> predict_ml -> explain_with_shap -> build_rag_queries
-> retrieve_policy_context -> evidence_guardrail
-> generate_grounded_response -> END
```

El state tipado separa solicitud cruda/validada, prediccion y probabilidad,
evidencia XAI, queries, evidencia documental, resultado del guardrail,
respuesta estructurada, citas, warnings, errores y traza de nodos. Una sola
instancia del embedder y del cliente Qdrant se reutiliza durante la ejecucion.
El champion y `explicar_solicitud()` se cargan sin reentrenar; las queries
proceden del helper determinista XAI->RAG existente.

Antes de llamar a Gemini, el guardrail exige resultados, URLs HTTPS de la
allowlist oficial, metadata completa y cobertura documental de al menos un
factor XAI. No adopta un threshold numerico a partir de la unica query OOD. Si
falla cualquier contrato, se devuelve una abstencion controlada y Gemini no se
invoca. La generacion usa el SDK moderno `google.genai`, temperatura 0 y JSON
schema. Provider y modelo se resuelven desde `GENAI_PROVIDER` y `GENAI_MODEL`;
el default free-tier-oriented es `gemini` / `gemini-3.5-flash-lite`. El prompt separa evidencia del
modelo, contexto normativo y warnings; prohibe causalidad, reglas o fuentes
inventadas, y limita las citas a identificadores asignados por el guardrail.

La capa de proveedor desactiva los retries internos del SDK y aplica como
maximo tres intentos controlados. Solo reintenta `503/UNAVAILABLE`, respuestas
no parseables y JSON truncado, con backoff exponencial de 1 y 2 segundos. Los
errores permanentes —por ejemplo autenticacion 4xx o schema invalido— fallan en
el primer intento. El state registra `generation_attempts`, un error tipado por
intento, `provider_status` y `generation_status`. Agotar los retries produce
`provider_unavailable`, sin citas ni explicacion grounded ficticia.

La prueba real del 1 de septiembre de 2026 proceso una aplicacion completa,
obtuvo prediccion `denied` con probabilidad positiva 0.208112, recupero 15
fuentes/chunks y supero el guardrail. Gemini devolvio `503 UNAVAILABLE` en dos
intentos por alta demanda, por lo que el artifact conserva honestamente una
abstencion controlada, no una explicacion inventada. La run registrada es
[`langgraph_grounded_generation_v1`](https://dagshub.com/jorgesialer/Proyecto-mle-riesgo.mlflow/#/experiments/0/runs/708a9858337b4d1f9bb7cdcfeccc3ecc).

Tras incorporar resiliencia se hizo una unica ejecucion live adicional. ML,
XAI, retrieval y guardrail finalizaron correctamente, pero los tres intentos
del proveedor devolvieron el mismo JSON truncado. El resultado fue
`generation_status=provider_unavailable`; no se creo un artifact de exito ni
la run `langgraph_grounded_generation_v2`. Esto es un fallo temporal del
proveedor, distinto de un fallo del workflow o de evidencia insuficiente.

El modelo anterior `gemini-2.5-flash` se conserva solo como antecedente: las
pruebas live presentaron `503 UNAVAILABLE` y JSON truncado por alta demanda.
Una unica validacion posterior con el nuevo default Flash-Lite recibio
`404 NOT_FOUND`: la API indico que el modelo ya no estaba disponible para
usuarios nuevos. Al ser un error permanente, no hubo retry, cambio silencioso
de modelo, artifact de exito ni run V2. La documentacion publica aun enumera
structured output y free tier para ese ID, pero la disponibilidad efectiva
depende de la cuenta/proyecto del proveedor.

El default actual es `gemini-3.5-flash-lite`, documentado como estable,
compatible con structured outputs y disponible en free tier. No cambia el
schema, evidencia, retries ni guardrails.

La unica validacion live con este default obtuvo JSON estructurado en el primer
intento, pero el modelo declaro citations como `S10`, `S13`, etc. en vez del
contrato exacto `[S#]`. El validador marco `output_validation_failed` y aplico
abstencion sin fuentes. No se creo `grounded_success_example.json` ni la run
`langgraph_grounded_generation_v2`; no se relajo el contrato ni se hizo otra
llamada.

La capa actual normaliza de forma determinista `S3`, `[S3]`, lowercase y
espacios al formato `[S3]` antes del allowlist, tanto en la lista declarada como
en texto libre. Strings que no son IDs `S` + digitos no se corrigen. La
telemetria `citation_normalization_count` registra cada ajuste sin convertirlo
en error cuando la fuente existe.

La unica validacion live posterior fue exitosa en el primer intento. Publico 11
fuentes, todas presentes en la evidencia recuperada, uso lenguaje no causal y
guardo `artifacts/langgraph/grounded_success_example.json`. En esa respuesta el
modelo ya emitio formato canonico, por lo que el contador fue 0. Run DagsHub:
[`langgraph_grounded_generation_v2`](https://dagshub.com/jorgesialer/Proyecto-mle-riesgo.mlflow/#/experiments/0/runs/3dc4691c60224b4ba59f3f1ed75b10c6).

Artifacts: `artifacts/langgraph/grounded_example.json`,
`input_application.json`, `source_metadata.json`, `graph_config.json`,
`prompt_template.txt`, `generation_metadata.json` y `run_metadata.json`.

Configuracion local, sin guardar credenciales en artifacts ni tracking:

```ini
GENAI_PROVIDER=gemini
GENAI_MODEL=gemini-3.5-flash-lite
GOOGLE_API_KEY=<configurar solo en .env o environment>
```

Puede copiarse `.env.example`; un `GENAI_MODEL` del environment reemplaza el
default. Solo Gemini esta habilitado en esta fase y nunca se hace fallback a
otro proveedor o modelo.

### MCP: exposicion estandarizada de capacidades V2

`src/mcp_server_v2.py` expone las capacidades existentes sin duplicar su
logica. La referencia del Modulo 4 del curso usa el SDK oficial `mcp`, tools
decoradas y transporte `stdio`; este proyecto conserva ese transporte local,
pero adopta la API vigente `MCPServer` de `mcp==2.1.1`, outputs estructurados y
un lifespan tipado. `stdio` evita abrir un puerto, autenticacion HTTP o
infraestructura adicional para esta entrega local.

El servidor publica exactamente cuatro tools read-only:

| Tool | Entrada | Salida |
|---|---|---|
| `predict_credit_application` | objeto `application` con las 16 variables crudas V2 | clase, label, probabilidad y modelo/version |
| `explain_credit_application` | la misma solicitud valida | evidence package SHAP local, top factors y warning no causal |
| `retrieve_credit_policy` | `query` y `top_k` entre 1 y 10 | chunks, scores y metadata oficial |
| `analyze_credit_application` | la misma solicitud valida | resultado completo del LangGraph existente, incluidos XAI, RAG, guardrail, generacion, fuentes y estados |

Los tres contratos de solicitud rechazan campos extra, target y atributos
`audit_only`. El lifecycle carga de forma perezosa y reutiliza una sola
instancia del champion/resumen, embedder, cliente Qdrant y grafo por sesion
MCP. La configuracion GenAI se carga solo cuando se solicita el workflow
completo, desde `.env` o el environment. Las respuestas son JSON estructurado;
no contienen API keys, rutas
locales ni objetos Python internos. Los errores internos se convierten en
mensajes controlados.

LangGraph sigue siendo el orquestador interno determinista que ejecuta
ML -> XAI -> RAG -> guardrail -> Gemini. MCP no es un segundo agente ni un
segundo workflow: es la interfaz estandar para invocar por separado tres
capacidades o el workflow completo.

Inicio local por `stdio`:

```bash
uv sync
uv run python -m src.mcp_server_v2
```

Configuracion equivalente de un cliente MCP:

```json
{
  "mcpServers": {
    "credit-approval-v2": {
      "command": "uv",
      "args": ["run", "python", "-m", "src.mcp_server_v2"]
    }
  }
}
```

Manifest, metadata y ejemplos reproducibles se generan sin una nueva llamada
live a Gemini:

```bash
uv run python -m src.mcp_server_v2 --generate-artifacts
```

Los artifacts quedan en `artifacts/mcp/`. El ejemplo de analisis reproduce el
output Gemini ya validado a traves del LangGraph real para comprobar el
contrato MCP sin introducir una llamada externa en esta fase.

## 6. Ejecucion

Desde la raiz del repositorio y con el entorno del proyecto activo:

```bash
# Sincronizar el entorno canonico desde pyproject.toml y uv.lock
uv sync

# Ejecutar tests
uv run python -m unittest discover -s tests -v

# Entrenar, validar, evaluar y persistir el baseline V1
uv run python -m src.entrenamiento

# Ejecutar el benchmark V2 completo con MLflow
uv run python -m src.entrenamiento_v2

# Generar y registrar XAI del champion existente, sin reentrenamiento
uv run python -m src.xai_v2 --sample-size 1000 --top-n 10

# Adquirir, indexar y evaluar el corpus RAG oficial
uv run python -m src.rag_v2

# Ejecutar ML -> XAI -> RAG -> guardrail -> Gemini y registrar la run
uv run python -m src.langgraph_v2 --backend dagshub

# Iniciar el servidor MCP local por stdio
uv run python -m src.mcp_server_v2

# Simular inferencia con un perfil crudo (V1)
uv run python -m src.prediccion
```

`pyproject.toml` y `uv.lock` son las fuentes canonicas de dependencias. El
proyecto no mantiene un `requirements.txt` paralelo.

`prediccion.py` recibe las variables categoricas originales; no requiere
columnas one-hot ni un scaler externo.

## 7. Estructura relevante

- `data/`: dataset crudo y archivos historicos locales.
- `src/preprocesamiento.py`: contrato de variables y construccion del pipeline.
- `src/entrenamiento.py`: split, CV, evaluacion y persistencia.
- `src/prediccion.py`: inferencia con el pipeline unico.
- `tests/`: pruebas del contrato de preprocesamiento e inferencia.
- `src/construir_dataset_v2.py`: extraccion, filtros, sampling y metadata HMDA.
- `src/preprocesamiento_v2.py`: feature engineering, split y pipeline V2.
- `src/entrenamiento_v2.py`: benchmark, robustness, tuning y champion V2.
- `src/mlflow_utils.py`: configuracion MLflow local/DagsHub.
- `src/xai_v2.py`: Tree SHAP global y evidencia local reusable del champion V2.
- `src/rag_v2.py`: adquisicion, chunking, Qdrant, retrieval y evaluacion RAG.
- `src/langgraph_v2.py`: orquestacion determinista y Gemini grounded fail-closed.
- `src/mcp_server_v2.py`: servidor MCP stdio con cuatro tools V2 estructuradas.
- `data/rag_sources/`: snapshots oficiales y manifest criptografico del corpus.
- `data/rag_eval_queries.json`: evaluation set manual de retrieval.
- `docs/HMDA_V2_DATA_DICTIONARY.md`: contrato de variables del Dataset V2.
- `artifacts/pipeline_rf_baseline.pkl`: pipeline evaluado y persistido (V1).
- `artifacts/pipeline_champion_v2.pkl`: pipeline champion persistido (V2).
- `artifacts/benchmark_v2_summary.json`: resumen de metricas y runs del benchmark V2.
- `notebooks/`: evidencia historica de experimentacion V1.

Los notebooks no son la fuente canonica del pipeline actual y sus resultados
anteriores no deben atribuirse al artefacto saneado.

## 8. Alcance y limitaciones

- El modelo predice aprobaciones historicas, no impago ni capacidad causal de
  repago.
- El benchmark V2 realiza una comparacion y tuning acotados; no constituye una
  busqueda exhaustiva de hiperparametros.
- La integracion Gemini de `src/prediccion.py` es un componente legado de V1. No constituye
  explicabilidad basada en evidencia del modelo. Su prompt identifica que el
  resultado proviene de ML y prohibe atribuir causalidad sin evidencia.
- El tracking del benchmark V2 y de XAI usa MLflow/DagsHub.
- La capa XAI describe asociaciones internas del champion mediante SHAP. No
  valida causalidad ni sustituye una auditoria de equidad.
- RAG/Qdrant recupera extractos, pero no determina elegibilidad ni sustituye
  la lectura de la fuente oficial; el corpus es deliberadamente acotado.
- El retriever aun no tiene threshold de abstencion. La unica query OOD obtuvo
  menor score que todas las queries in-domain, pero no basta para calibrarlo.
- LangGraph/Gemini V2 sintetiza evidencia, no toma decisiones crediticias. La
  disponibilidad de Gemini es externa; `503` o JSON incompleto tras tres
  intentos producen abstencion controlada.
- Las citas pasan una validacion de pertenencia al retrieval, pero esto no
  garantiza que la sintesis sea juridicamente correcta ni exhaustiva.
- MCP esta implementado como interfaz local `stdio`; no incorpora control de
  acceso de red, rate limiting ni aislamiento multi-tenant. Docker sigue fuera
  del alcance.
- Los atributos demograficos pueden reproducir sesgos presentes en decisiones
  historicas y requieren una evaluacion de equidad separada.

## 9. Conclusiones

- **Eficacia del Enfoque Hibrido (ML + GenAI):** La combinacion de un modelo clasico de Machine Learning (`RandomForestClassifier`) con un modelo de lenguaje fundacional (`Gemini 2.5 Flash`) demostro ser una arquitectura viable para el sector financiero, integrando la prediccion probabilistica del baseline con una sintesis descriptiva para comites de credito. En fases posteriores V2 se incorporara explicabilidad formal basada en evidencia (XAI) para sustentar los reportes.

- **Robustez y Estabilidad de la Prediccion:** El modelo predictivo alcanzo en el baseline saneado un F1-Score medio de 0.933760 (+/- 0.015653) bajo validacion cruzada estratificada de 5 folds y un F1-Score de 0.912752 en el holdout final (ROC-AUC de 0.944130 y PR-AUC de 0.903451). Estos resultados confirman que el pipeline unificado y fold-safe mitiga el desbalance mediante `class_weight='balanced'` y previene data leakage.

- **Determinacion Contextual Coherente:** Mediante la inferencia simulada en `src/prediccion.py`, el modelo de lenguaje contextualiza los factores financieros respetando las variables de entrada proporcionadas por el pipeline de datos, manteniendo un tono corporativo y reconociendo explicitamente su naturaleza descriptiva sin atribuir causalidad no demostrada.

- **Delimitacion del Alcance Operativo:** Para la version 1.0.0 saneada, el proyecto cumple exitosamente con el objetivo de automatizar de punta a punta el pipeline de datos, el entrenamiento parametrizado, la inferencia y una suite de pruebas unitarias bajo `uv`. Se establece asi una base tecnica solida y reproducible para el desarrollo de las fases V2 (nuevas variables, MLflow, XAI, RAG y orquestacion con LangGraph).
