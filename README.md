# Credit Approval AI System V2

Sistema de Machine Learning + Generative AI para modelar decisiones históricas de aprobación crediticia sobre solicitudes hipotecarias HMDA 2023.

La **Versión 2 (V2)** evoluciona el baseline educativo previo hacia un sistema de nivel de producción que combina:

- **Machine Learning**: Clasificación tabular con **CatBoost** como modelo champion (`catboost_tuned_02`).
- **Seguimiento y Registro**: Experiment tracking y Model Registry remoto en **MLflow / DagsHub**.
- **Explicabilidad (XAI)**: Atribución de factores locales y globales en escala log-odds mediante **Tree SHAP**.
- **Recuperación Normativa (RAG)**: Búsqueda vectorial densa sobre normativa oficial CFPB y Fannie Mae con **Qdrant**.
- **Orquestación Determinista**: Grafo lineal tipado y guardrails fail-closed con **LangGraph**.
- **Generación Grounded**: Síntesis analítica estructurada y libre de alucinaciones con **Gemini 3.5 Flash-Lite**.
- **Interoperabilidad (MCP)**: Exposición estandarizada de capacidades del sistema vía **Model Context Protocol (MCP)**.

La **Versión 1 (V1)**, basada en un dataset educativo de 5,000 filas y un baseline Random Forest, se conserva intacta como referencia histórica en la release [`v1.0.0`](https://github.com/jorgesialer/Proyecto-mle-riesgo/releases/tag/v1.0.0).

---

## Enlaces de Evidencia y Tracking Remoto

| Recurso | Enlace |
| :--- | :--- |
| **GitHub Release actual** | [Release v2.0.1](https://github.com/jorgesialer/Proyecto-mle-riesgo/releases/tag/v2.0.1) |
| **Repositorio DagsHub** | [DagsHub: Proyecto-mle-riesgo](https://dagshub.com/jorgesialer/Proyecto-mle-riesgo) |
| **DagsHub Experiments** | [DagsHub Experiments Dashboard](https://dagshub.com/jorgesialer/Proyecto-mle-riesgo/experiments) |
| **MLflow UI** | [Servidor MLflow Tracking](https://dagshub.com/jorgesialer/Proyecto-mle-riesgo.mlflow) |
| **Model Registry** | [Modelo registrado: `credit-approval-v2`](https://dagshub.com/jorgesialer/Proyecto-mle-riesgo.mlflow/#/models/credit-approval-v2) |
| **Run: Benchmark ML Champion** | [Run `champion_catboost_final` (94964b60)](https://dagshub.com/jorgesialer/Proyecto-mle-riesgo.mlflow/#/experiments/0/runs/94964b600522497a8cc306401c09fd79) |
| **Run: Explicabilidad SHAP** | [Run `xai_catboost_champion_v2` (f8e8fc29)](https://dagshub.com/jorgesialer/Proyecto-mle-riesgo.mlflow/#/experiments/0/runs/f8e8fc291f8749a9b18634c5530ee054) |
| **Run: Recuperación RAG / Qdrant** | [Run `rag_qdrant_retrieval_v2` (1e98c472)](https://dagshub.com/jorgesialer/Proyecto-mle-riesgo.mlflow/#/experiments/0/runs/1e98c47241ff449bb5bcb41cb79caf9c) |
| **Run: LangGraph + Gemini Grounded** | [Run `langgraph_grounded_generation_v2` (3dc4691c)](https://dagshub.com/jorgesialer/Proyecto-mle-riesgo.mlflow/#/experiments/0/runs/3dc4691c60224b4ba59f3f1ed75b10c6) |

---

## 1. Arquitectura del Sistema V2

El sistema sigue un principio fundamental de gobernanza técnica: **Machine Learning predice, Model Explainability proporciona evidencia, RAG recupera contexto normativo oficial, el LLM sintetiza explicaciones grounded y LangGraph orquesta el flujo.** El LLM nunca sustituye al modelo predictivo.

```text
[ Solicitud de Crédito: 16 predictores ]
                   │
                   ▼
┌───────────────────────────────────────┐
│ 1. Validación de Entrada & Contrato   │ ──> Rechaza si contiene target o audit-only
└──────────────────┬────────────────────┘
                   │
                   ▼
┌───────────────────────────────────────┐
│ 2. Predicción ML (CatBoost Champion)  │ ──> Probabilidad y decisión binaria (LoanApproved)
└──────────────────┬────────────────────┘
                   │
                   ▼
┌───────────────────────────────────────┐
│ 3. Atribución Local (Tree SHAP)       │ ──> Evidence package: factores positivos/negativos en log-odds
└──────────────────┬────────────────────┘
                   │
                   ▼
┌───────────────────────────────────────┐
│ 4. Generación de Consultas Normativas │ ──> Queries deterministas basadas en factores SHAP
└──────────────────┬────────────────────┘
                   │
                   ▼
┌───────────────────────────────────────┐
│ 5. Búsqueda Vectorial (Qdrant + RAG)  │ ──> Recupera fragmentos de guías CFPB y Fannie Mae
└──────────────────┬────────────────────┘
                   │
                   ▼
┌───────────────────────────────────────┐
│ 6. Guardrail de Evidencia             │ ──> Verifica HTTPS allowlist, metadata completa y cobertura XAI
└──────────────────┬────────────────────┘     (Abstención controlada si falla)
                   │
                   ▼
┌───────────────────────────────────────┐
│ 7. Síntesis Grounded (Gemini 3.5)     │ ──> JSON estructurado con citas [S#] validadas y sin causalidad
└──────────────────┬────────────────────┘
                   │
                   ▼
┌───────────────────────────────────────┐
│ 8. Interfaz MCP (Stdio Transport)     │ ──> 4 herramientas tipadas para clientes externos
└───────────────────────────────────────┘
```

---

## 2. Dataset HMDA 2023 y Feature Engineering

V2 utiliza datos reales del **HMDA 2023 One Year National Loan-Level Dataset** oficial de CFPB/FFIEC.

- **Target conceptual:** `LoanApproved` (`1` = originada o aprobada no aceptada; `0` = denegada). No modela default ni capacidad causal de repago.
- **Muestra canónica:** 50,000 solicitudes obtenidas mediante reservoir sampling determinista (`random_state=42`) sobre la población hipotecaria estándar (préstamos 1-4 unidades, primer gravamen, convencionales, compra o refinanciamiento).
- **Separación de variables sensibles (`audit_only`):** Las variables demográficas (`applicant_age`, `derived_race`, `derived_ethnicity`, `derived_sex`) están aisladas exclusivamente para auditoría posterior y están estrictamente bloqueadas del entrenamiento, inferencia y explicabilidad.
- **Pipeline de variables:** 16 predictores crudos procesados con imputación por mediana/moda, codificación one-hot e ingeniería de ratios contractuales (Loan-to-Income, PropertyValue-to-Income, bandas HMDA normalizadas de DTI y flags de amortización).

Consulte la documentación detallada en:
- [Diccionario de Datos V2](docs/HMDA_V2_DATA_DICTIONARY.md)
- [Especificación Técnica V2](docs/V2_SPEC.md)
- [Ficha del Modelo](model-card.md)

---

## 3. Resultados Principales del Benchmark ML

El benchmark V2 evaluó 11 configuraciones competitivas (Random Forest, XGBoost y CatBoost) mediante validación cruzada estratificada de 5 folds sobre 40,000 registros de entrenamiento, reservando 10,000 registros aislados para holdout final.

### Champion Seleccionado: `catboost_tuned_02`

- **Familia:** CatBoost (`iterations=450`, `depth=7`, `learning_rate=0.04`)
- **Conjunto de variables:** `full` (16 predictores + ratios financieros derivados)

| Métrica | CV 5-Folds (Training) | Holdout Final (Test) |
| :--- | :---: | :---: |
| **Precision** | 0.883045 ± 0.002088 | 0.881443 |
| **Recall** | 0.966567 ± 0.001097 | 0.965606 |
| **F1-Score** | **0.922920 ± 0.001618** | **0.921607** |
| **ROC-AUC** | 0.878539 ± 0.005286 | 0.874957 |
| **Average Precision (PR-AUC)** | 0.952849 ± 0.002374 | 0.951897 |

Matriz de confusión en holdout (10,000 registros):
- **Verdaderos Negativos (TN):** 1,196 | **Falsos Positivos (FP):** 1,012
- **Falsos Negativos (FN):** 268 | **Verdaderos Positivos (TP):** 7,524

> **Interpretación comparativa:** CatBoost superó a `xgb_baseline_full` (CV F1: 0.922287) por una diferencia marginal (Delta F1 aproximado 0.0006), menor que la desviación entre folds (0.0016). Ambas familias de gradient boosting exhibieron alta solidez. Los resultados de V1 y V2 **no son directamente comparables** al utilizar fuentes, esquemas y poblaciones distintas.

---

## 4. Componentes de la Capa GenAI

### Explicabilidad del Modelo (Tree SHAP)
- **Implementación:** [`src/xai_v2.py`](src/xai_v2.py) aplica Tree SHAP directamente sobre el pipeline champion.
- **Gobernanza:** Agrupa columnas one-hot por feature de negocio para eliminar sesgos de cardinalidad (`mean(abs(grouped_shap))`) y genera un `evidence_package` estructurado con etiquetas HMDA oficiales legibles.
- **Artefactos:** Visualizaciones globales en [`artifacts/xai/shap_summary.png`](artifacts/xai/shap_summary.png) y [`artifacts/xai/shap_bar.png`](artifacts/xai/shap_bar.png).

### Recuperación Normativa RAG (Qdrant)
- **Implementación:** [`src/rag_v2.py`](src/rag_v2.py) gestiona un almacén vectorial local embebido en Qdrant.
- **Corpus Oficial:** 139 fragmentos semánticos procedentes de 5 documentos oficiales con hash SHA-256 verificado (CFPB HMDA 2023 Reference Chart y guías B2 y B3 de Fannie Mae Selling Guide).
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (384 dimensiones, distancia coseno).

### Orquestación y Generación Grounded (LangGraph + Gemini)
- **Implementación:** [`src/langgraph_v2.py`](src/langgraph_v2.py) compila un `StateGraph` determinista y lineal.
- **Guardrail de evidencia:** Comprueba procedencia HTTPS en allowlist oficial (`files.consumerfinance.gov`, `selling-guide.fanniemae.com`), completitud de metadatos y cobertura documental antes de contactar al proveedor de lenguaje.
- **Generación estructurada:** Invoca `gemini-3.5-flash-lite` bajo esquema JSON estricto (`temperature=0`), con normalización determinista de citas (`[S#]`) y abstención controlada *fail-closed* ante fallos de conectividad o evidencia insuficiente.

### Protocolo de Contexto de Modelo (MCP)
- **Implementación:** [`src/mcp_server_v2.py`](src/mcp_server_v2.py) utiliza el SDK oficial `mcp==2.1.1` mediante transporte `stdio`.
- **Herramientas publicadas:**
  1. `predict_loan_approval`: Inferencia probabilística con CatBoost Champion.
  2. `explain_loan_prediction`: Atribución local de factores con Tree SHAP.
  3. `retrieve_credit_policy`: Búsqueda vectorial semántica en el corpus de Qdrant.
  4. `analyze_loan_application`: Pipeline completo orquestado con LangGraph y Gemini.
- **Ciclo de vida:** Gestión perezosa y reutilizable de recursos mediante `MCPResourceManager` para evitar sobrecarga de memoria en sesiones continuas.

---

## 5. Cómo Ejecutar

El proyecto utiliza [`uv`](https://docs.astral.sh/uv/) como gestor canónico de dependencias y entornos virtuales.

### Sincronización del entorno y tests
```bash
# Instalar dependencias exactas
uv sync

# Ejecutar la suite completa de tests (96 tests)
uv run python -m unittest discover -s tests -v
```

### Ejecución de los módulos V2
```bash
# 1. Construir dataset V2 HMDA (50,000 filas)
uv run python -m src.construir_dataset_v2

# 2. Ejecutar benchmark y registrar runs en MLflow
uv run python -m src.entrenamiento_v2

# 3. Generar explicabilidad SHAP del champion
uv run python -m src.xai_v2 --sample-size 1000 --top-n 10

# 4. Construir índice y evaluar RAG local con Qdrant
uv run python -m src.rag_v2 --skip-mlflow

# 5. Ejecutar workflow completo LangGraph + Gemini
uv run python -m src.langgraph_v2 --backend dagshub

# 6. Iniciar servidor MCP local (stdio)
uv run python -m src.mcp_server_v2
```

---

## 6. Estructura del Repositorio

```text
├── .env.example             # Plantilla de configuración de variables y llaves
├── pyproject.toml           # Definición de dependencias y configuración del proyecto
├── uv.lock                  # Lockfile canónico y reproducible
├── README.md                # Presentación general técnica del proyecto
├── model-card.md            # Ficha técnica detallada del modelo
├── docs/                    # Especificaciones y diccionarios técnicos
│   ├── V2_SPEC.md           # Especificación de arquitectura V2
│   └── HMDA_V2_DATA_DICTIONARY.md # Diccionario de variables del dataset V2
├── data/                    # Datasets y fuentes documentales
│   ├── hmda_2023_loan_approval_v2.csv # Muestra HMDA V2 (50k registros)
│   └── rag_sources/         # Snapshots oficiales CFPB / Fannie Mae
├── src/                     # Código fuente de producción
│   ├── construir_dataset_v2.py # Ingesta y muestreo del dataset HMDA
│   ├── preprocesamiento_v2.py  # Feature engineering y pipeline sklearn
│   ├── entrenamiento_v2.py     # Benchmark ML y champion selection
│   ├── mlflow_utils.py         # Utilidades de tracking y registro MLflow
│   ├── xai_v2.py               # Explicabilidad Tree SHAP
│   ├── rag_v2.py               # Indexación y búsqueda semántica con Qdrant
│   ├── langgraph_v2.py         # Orquestación determinista y Gemini
│   └── mcp_server_v2.py        # Servidor MCP stdio con 4 herramientas
├── tests/                   # Suite de pruebas unitarias (96 tests)
└── artifacts/               # Artefactos versionados de soporte y evaluación
    ├── benchmark_v2_summary.json
    ├── xai/                 # Metadatos, importancias y gráficos SHAP
    ├── rag/                 # Chunks y evaluación de retrieval
    ├── langgraph/           # Ejemplos de ejecución y configuración del grafo
    └── mcp/                 # Manifiestos de herramientas y metadatos MCP
```

---

## 7. Alcance y Limitaciones

- **Carácter del target:** El modelo predice la decisión histórica de aprobación (`LoanApproved`), no la solvencia moral, el riesgo de crédito futuro ni la probabilidad de default.
- **No causalidad:** Las atribuciones de SHAP reflejan asociaciones estadísticas con el margen del estimador; no deben interpretarse como causas reales del otorgamiento crediticio.
- **Corpus normativo acotado:** El componente RAG recupera guías oficiales de referencia, pero no reemplaza el criterio humano ni el análisis jurídico-financiero institucional.
- **Comportamiento fail-closed:** Ante indisponibilidad de APIs externas, citas no mapeadas o metadatos insuficientes, el sistema se abstiene de forma controlada sin emitir respuestas inventadas.
- **Aislamiento de atributos demográficos:** Las variables sensibles no intervienen en el modelo para evitar discriminación directa, aunque se reconoce que los datos históricos pueden reflejar sesgos sistémicos inherentes al proceso de originación.
- **Entorno de MCP:** El servidor MCP opera mediante `stdio` local; no incluye capa de red multi-tenant ni autenticación HTTP delegada. Docker permanece fuera del alcance actual.

---

## 8. Conclusiones

1. **Eficacia de la Arquitectura V2:** La separación estricta de responsabilidades (**ML predice, XAI explica, RAG contextualiza, LLM sintetiza y LangGraph orquesta**) demostró ser una solución técnicamente sólida, auditable y gobernable para el sector financiero.
2. **Solidez Predictiva:** El champion CatBoost alcanzó un F1-Score de **0.9229** en CV y **0.9216** en holdout final sobre 50,000 créditos reales de HMDA 2023, superando al baseline Random Forest y a XGBoost en estabilidad.
3. **Explicabilidad Libre de Fuga:** Tree SHAP proporcionó explicaciones locales matemáticamente rigurosas sobre el espacio de margen del modelo, excluyendo estructuralmente variables protegidas.
4. **Prevención de Alucinaciones:** El diseño fail-closed del grafo garantizó que ninguna respuesta sea generada sin respaldo en la evidencia recuperada, normalizando citas e impidiendo la afirmación de relaciones causales ficticias.
5. **Estandarización Abierta:** La interfaz MCP permite a plataformas externas y agentes de IA interactuar con el modelo y el corpus de conocimiento sin acoplarse a la implementación interna.

---

## 9. Historial de Versiones

### v2.0.1 (Versión Actual)
- Reorganización estructural del `README.md` orientada a destacar la arquitectura V2, resultados del benchmark y enlaces remotos de DagsHub/MLflow.
- Resumen y reubicación de la versión histórica V1.
- Actualización integral de conclusiones y métricas vigentes.

### v2.0.0
- Release técnica de la arquitectura V2 completa.
- Incorporación del dataset HMDA 2023 (50,000 filas) y CatBoost Champion (`catboost_tuned_02`).
- Capa XAI con Tree SHAP, RAG local con Qdrant y orquestación con LangGraph + Gemini 3.5 Flash-Lite.
- Implementación de 4 herramientas mediante el Model Context Protocol (MCP stdio).
- Cobertura total de 96 pruebas unitarias exitosas.

### v1.0.0
- Release histórica inicial del proyecto.
- Basada en un dataset educativo sintético de 5,000 registros con Random Forest como baseline.
- Saneamiento del pipeline para evitar fuga de datos (*data leakage*) mediante `ColumnTransformer` y validación cruzada estratificada.
- Preservada como referencia histórica en el tag [`v1.0.0`](https://github.com/jorgesialer/Proyecto-mle-riesgo/releases/tag/v1.0.0).