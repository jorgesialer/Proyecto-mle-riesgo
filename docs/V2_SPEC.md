# Credit Risk AI System — Version 2.0

## 1. Project objective

Develop version 2.0 of the existing Machine Learning Engineering project
for loan approval prediction.

The prediction target remains:

`LoanApproved`

The project must preserve Machine Learning as the core decision engine while
incorporating Generative AI capabilities that improve explainability,
retrieval of domain knowledge and interaction with the prediction system.

---

## 2. Version 1 baseline

Version 1 currently contains:

- loan approval classification,
- preprocessing pipeline,
- Random Forest classifier,
- offline evaluation,
- cross-validation,
- reusable Python scripts,
- notebook-based experimentation,
- model documentation,
- Gemini integration for generating a textual explanation of the prediction.

Version 2 must evolve this architecture rather than replace it.

## 2.1 V1 technical debt to resolve before V2

Before implementing new V2 functionality, the V1 baseline must be corrected
and made reproducible.

Required corrections:

1. Split raw data into training and test sets before fitting any data-dependent
   preprocessing transformation.

2. Fit imputers, encoders and any required transformations only on training
   data.

3. Use a single scikit-learn Pipeline / ColumnTransformer so preprocessing and
   the estimator are part of the same reproducible artifact.

4. Cross-validation must evaluate the complete pipeline and must be fold-safe.

5. Preserve a final holdout test set that is not used for model selection,
   tuning or demonstrations.

6. Evaluate exactly the model configuration that is persisted as the baseline.

7. Correct documentation semantics:
   `LoanApproved` represents historical loan approval, not observed loan default.

8. Remove unsupported claims about model explainability, hallucination
   elimination or default-risk estimation.

9. Replace the current Gemini explanation mechanism with evidence-based model
   explainability before using an LLM to justify predictions.

10. Resolve dependency and environment reproducibility before adding new
    V2 technologies.
---

## 3. Main V2 goals

### Machine Learning

1. Improve the dataset representation with additional meaningful financial
   variables.

2. Perform stronger feature engineering.

3. Build a reproducible preprocessing and training pipeline.

4. Keep Random Forest as a baseline.

5. Compare multiple algorithms studied in the specialization, such as:

   - Random Forest
   - XGBoost
   - LightGBM
   - CatBoost

6. Track experiments, parameters, metrics and artifacts using MLflow.

7. Select a champion model based on appropriate classification metrics.

8. Add model explainability so that downstream GenAI components receive
   evidence from the ML model rather than inventing explanations.

---

## 4. Generative AI goals

The GenAI layer must complement the Machine Learning system.

Planned components:

1. Credit-policy knowledge base.

2. Embeddings and vector retrieval.

3. Qdrant vector database.

4. RAG pipeline for retrieving relevant policy information.

5. LangGraph agent.

6. Tools that allow the agent to:

   - request a credit-risk prediction,
   - obtain model explanation information,
   - retrieve relevant credit policies.

7. The LLM should synthesize the information into a grounded explanation for
   a credit analyst.

8. MCP / FastMCP may later expose these capabilities as reusable tools.

---

## 5. Intended architecture

User / credit application

        ↓

Feature engineering

        ↓

Machine Learning model

        ↓

Prediction + probability

        ↓

Model explainability

        ↓

        ┌─────────────────────┐
        │                     │
        ▼                     ▼

ML evidence          Credit-policy RAG

        │                     │
        └──────────┬──────────┘
                   ▼

            LangGraph agent

                   ↓

                  LLM

                   ↓

       Grounded analyst explanation

---

## 6. Dataset V2

The approved source is the official CFPB/FFIEC **HMDA 2023 One Year National
Loan-Level Dataset**, with a data freeze date of May 19, 2025. V1 remains an
independent historical baseline; no joins or synthetic enrichment are made
between V1 and V2.

The binary target remains conceptually unchanged:

- `action_taken = 1` (originated) -> `LoanApproved = 1`
- `action_taken = 2` (approved but not accepted) -> `LoanApproved = 1`
- `action_taken = 3` (denied) -> `LoanApproved = 0`

All other actions are excluded and `action_taken` is removed immediately
after target construction.

The V2 population is limited to 2023 closed-end, non-reverse,
non-business/commercial applications for properties with one to four units.
The target dataset is an unbalanced, reproducible 50,000-row reservoir sample
using `random_state=42`; no SMOTE or artificial 50/50 balancing is applied.

The prediction moment is immediately before the final decision, after
underwriting inputs are available. The final raw dataset contains 16 candidate
predictors covering income, requested loan, DTI, CLTV, property value, loan
and property characteristics, and proposed non-amortizing terms. It also
contains four `audit_only` columns (`applicant_age`, `derived_race`,
`derived_ethnicity`, `derived_sex`) that are explicitly prohibited from
entering model predictors.

Identifiers, geography, protected raw attributes, denial reasons, AUS
results, purchaser type, interest/rate spread, pricing/cost fields and other
post-decision fields are blacklisted. The full roles, source names, missing
codes and leakage notes are documented in `docs/HMDA_V2_DATA_DICTIONARY.md`.

The official national raw file is not stored. Construction streams the
filtered Data Browser CSV in gzip, applies remaining population filters and
reservoir sampling, and persists only the final 21-column sample plus
reproducibility metadata.

### Final schema audit

The 50,000-row sample removed three low-information variables:

- `negative_amortization`: constant (`2`) in all rows;
- `introductory_rate_period`: 93.214% structural missingness;
- `other_nonamortizing_features`: only 44 positive values (0.088%).

`total_units` is retained as categorical because 868 multi-unit applications
remain across four observed categories. Income, DTI, CLTV and property value
are also retained, but their missingness is materially higher for denied
applications. No missingness indicator is added at this stage.

### Approved feature engineering contract

Deterministic feature engineering produces Loan-to-Income,
PropertyValue-to-Income, Loan-to-Property-Value, loan term in years, a
canonical seven-band DTI category and a count of the surviving non-amortizing
contract flags. Invalid divisions produce missing values; they are not filled
until the training-only preprocessing pipeline is fitted.

The model input after deterministic feature engineering has 20 columns: nine
numeric and eleven categorical. Audit-only attributes and `LoanApproved` are
excluded through an explicit whitelist before splitting.

The initial evaluation design uses an 80/20 stratified train/final-holdout
split. Cross-validation and model selection operate strictly on the 80%
training partition (40,000 samples across 5 stratified folds). Feature
engineering, median/mode imputation and `OneHotEncoder(handle_unknown="ignore")`
are composed inside the sklearn pipeline so every CV fold remains leakage-safe.
The V2 benchmark has been executed comparing Random Forest, XGBoost, and
CatBoost; `catboost_tuned_02` was selected as champion based on CV F1 score
and evaluated once on the final holdout. All benchmark runs, metrics, and
artifacts are persisted in `artifacts/benchmark_v2_summary.json` with remote
tracking and model registry on DagsHub.

### Data provenance requirement

Every V2 variable must have a documented source and availability timestamp.

Credit-history variables must represent information available strictly before
the loan application decision.

No variable derived from post-decision outcomes may be used as a predictor.

This approved dataset uses no synthetic rows or synthetic raw variables.

Before implementation, each candidate variable must be classified as:

- original observed variable,
- derived feature,
- externally sourced variable,
- synthetic variable.

Synthetic variables require explicit justification and documented generation
assumptions.
---

## 7. ML evaluation

Candidate metrics:

- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC
- Confusion Matrix

Accuracy should not be the only model-selection criterion.

The final metric strategy will be defined after analyzing class balance and
business implications.

---

## 8. MLflow

MLflow should track, when applicable:

- model type,
- hyperparameters,
- preprocessing configuration,
- metrics,
- confusion matrix,
- feature importance / explainability artifacts,
- trained model artifact.

A champion model should be selected based on experimental evidence.

---

## 9. GenAI principles

The LLM must not fabricate the reason behind an ML prediction.

Instead:

ML model
→ prediction

Explainability layer
→ model evidence

RAG
→ policy evidence

LLM
→ grounded natural-language explanation
### Orchestration strategy

The initial V2 orchestration will prefer a deterministic LangGraph workflow
over an autonomous ReAct loop.

The intended sequence is:

prediction
→ model evidence
→ policy retrieval
→ grounded synthesis

Agentic tool selection may be evaluated later only if it provides a clear
benefit over the deterministic workflow.
---

## 10. Technologies intentionally excluded for now

Do not add technologies only for portfolio appeal.

For the first V2 implementation, avoid adding technologies that were not
covered or justified by the specialization.

Possible future additions can be evaluated after V2 is functional.

---

## 11. Development phases

### Phase 1
Dataset V2 design.

### Phase 2
Feature engineering and preprocessing pipeline.

### Phase 3
ML baseline and model benchmarking.

### Phase 4
MLflow tracking and champion model.

### Phase 5
Model explainability. **Implemented:** the persisted CatBoost champion is
explained with Tree SHAP without retraining. Global evidence uses a reproducible
1,000-row sample from the training partition, and local evidence packages map
numeric, derived and one-hot columns back to business-readable features.
Global business-feature importance first sums transformed SHAP contributions
within each row and only then computes mean absolute contribution across rows.
Relevant HMDA enumerations use official human-readable labels while preserving
their raw codes; unmapped codes are explicitly marked unknown.
Audit-only attributes remain structurally excluded. SHAP values describe model
contributions in raw-margin space and must not be interpreted as causal effects.

### Phase 6
Credit-policy RAG with Qdrant. **Implemented:** `src/rag_v2.py` acquires five
official, scoped sources (one CFPB HMDA 2023 reference and four Fannie Mae
Selling Guide sections covering income, DTI, LTV and CLTV), records URL,
version/date, retrieval timestamp and SHA-256, and extracts text reproducibly.
PDF pages and HTML sections are chunked deterministically at 1,000 characters
with 150-character overlap. Nested HTML blocks are deduplicated so a paragraph
inside a list item is indexed once. The corrected snapshot contains 139 chunks from 65
sections.

Embeddings are generated locally with
`sentence-transformers/all-MiniLM-L6-v2` (384 dimensions, normalized) and
stored in the `credit_policy_v2` collection using Qdrant embedded/local mode
and cosine distance. No Docker or paid embedding API is required. Runtime
Qdrant storage is rebuildable and excluded from Git; source snapshots,
manifest, chunks and evaluation artifacts remain traceable.

The manual evaluation preserves 12 directed queries and adds four fixed cases:
ambiguous LTV/CLTV, multi-source income/debt, a less literal income paraphrase,
and an out-of-domain business-checking query. Directed source hit@5, source
recall@5, concept hit@5 and MRR@5 remain 1.0. The three additional in-domain
queries also score 1.0; the out-of-domain query is not rejected and returns a
top score of 0.363723, versus a minimum in-domain top score of 0.534716. No
threshold is adopted from a single OOD observation; abstention requires a
larger calibration set. These figures are not evidence of broad regulatory
coverage or answer-generation quality. XAI factors are converted to queries by
deterministic templates; audit-only attributes are excluded. This phase does
not call Gemini by itself; its APIs are reused by Phase 7.

### Phase 7
Deterministic LangGraph orchestration with grounded Gemini generation.
**Implemented:** `src/langgraph_v2.py` executes the fixed sequence
`validate_input -> predict_ml -> explain_with_shap -> build_rag_queries ->
retrieve_policy_context -> evidence_guardrail -> generate_grounded_response`.
It is a compiled linear `StateGraph`, not a ReAct agent. The typed state keeps
raw and validated applications, prediction/probability, XAI evidence, RAG
queries, retrieved policy evidence, sufficiency status, structured/final
responses, citations, warnings, errors and node trace.

The persisted CatBoost champion, `explicar_solicitud()` and
`construir_queries_desde_evidence()` are reused without retraining or
reimplementation. One local embedder and Qdrant client are reused per workflow
execution. Audit-only fields fail schema validation and cannot reach ML, XAI,
retrieval or the prompt.

The evidence guardrail is fail-closed. It requires non-empty retrieval,
complete source metadata, official HTTPS hosts and policy evidence for at
least one relevant XAI factor. It deliberately has no numeric score threshold
because the retrieval evaluation contains only one OOD query. Insufficient or
invalid evidence produces a controlled abstention without calling Gemini.
Generated citation identifiers are also checked against retrieved evidence;
unknown citations cause abstention.

Grounded synthesis uses the modern `google.genai` SDK, temperature zero and a
JSON schema. Provider and model are resolved from `GENAI_PROVIDER` and
`GENAI_MODEL`; the documented free-tier-oriented defaults are `gemini` and
`gemini-3.5-flash-lite`. Environment values override those defaults. Model evidence, policy
evidence and warnings are separate prompt sections. The prompt forbids causal
claims, invented rules/thresholds/sources, treating SHAP as direct probability
change, or attributing a historical model decision to retrieved policy.

Provider resilience is explicit and bounded. SDK-internal retries are disabled
and the workflow performs at most three calls, retrying only HTTP
`503/UNAVAILABLE`, unparseable responses and truncated JSON. Retry delays are
1 and 2 seconds (exponential factor 2). Permanent 4xx authentication errors
and schema validation errors are not retried. State records attempt count,
typed error details per attempt, final provider status and generation status.
Exhaustion produces `generation_status=provider_unavailable` and a fail-closed
response with no citations or fabricated grounded explanation.

The real integration on 2026-09-01 predicted `denied` with positive-class
probability 0.208112, retrieved 15 source chunks and passed the evidence
guardrail. Gemini returned `503 UNAVAILABLE` twice due temporary high demand;
the persisted sample therefore records the expected controlled abstention.
This validates the live failure path, not successful answer quality. DagsHub
run: `langgraph_grounded_generation_v1`, id
`708a9858337b4d1f9bb7cdcfeccc3ecc`.

One live validation was executed after adding retries. The deterministic ML,
XAI, RAG and guardrail stages succeeded, but all three provider responses were
truncated at the same JSON position. The workflow recorded
`provider_unavailable` and abstained. No `grounded_success_example.json` and no
`langgraph_grounded_generation_v2` run were created. This is classified as
provider unavailability, not workflow failure or insufficient policy evidence.

`gemini-2.5-flash` remains historical evidence: its live calls produced high
demand `503` responses and truncated JSON. One subsequent end-to-end execution
used the configured Flash-Lite default. The model's public documentation lists
structured outputs and the existing JSON-schema/temperature configuration is
compatible, but this account received permanent `404 NOT_FOUND` stating that
the model was unavailable to new users. The workflow correctly made one call,
did not retry, did not switch models, abstained and created neither a success
artifact nor the V2 DagsHub run. Public capability and free-tier listing do not
guarantee account-level availability.

API keys are read only by the Gemini SDK from the process environment. They are
not part of `GenAIConfig`, workflow state, prompts, artifacts or MLflow params;
known key values are defensively redacted from provider error messages.

The current default is `gemini-3.5-flash-lite`, which official model and
pricing documentation list as stable, structured-output capable and available
on the free tier. Provider, JSON schema, temperature, evidence contract,
retries and fail-closed behavior remain unchanged.

The single live validation with this default returned schema-valid JSON on the
first provider call. It nevertheless declared citation identifiers as `S10`,
`S13`, etc. rather than the exact `[S#]` evidence contract. Citation validation
therefore produced `output_validation_failed` and a controlled abstention with
no sources. No success artifact or V2 DagsHub run was created, and the contract
was not weakened or retried.

Citation syntax is now normalized deterministically before allowlist
validation. `S3`, `[S3]`, lowercase and surrounding spaces become `[S3]` in
both declared IDs and free text. Only the restricted `S` plus digits grammar is
accepted for normalization; other strings remain unchanged and fail normally.
`citation_normalization_count` is stored in state, artifacts and MLflow.

The single post-fix live validation succeeded on its first provider call. JSON
and citations passed validation, all 11 published sources existed in retrieved
policy evidence, and the response explicitly separated model association from
policy context and denied causal attribution. The model emitted canonical IDs
in this run, so the normalization count was 0. Artifact:
`artifacts/langgraph/grounded_success_example.json`. DagsHub run:
`langgraph_grounded_generation_v2`, id
`3dc4691c60224b4ba59f3f1ed75b10c6`.

### Phase 8
MCP integration. **Implemented:** `src/mcp_server_v2.py` uses the official
`mcp==2.1.1` Python SDK and the current `MCPServer` API. The course reference
also uses the official SDK, decorated tools and `stdio`; V2 keeps `stdio` but
updates the older eager-loading/string-output example to typed structured
outputs and a server lifespan.

Exactly four read-only tools are exposed: historical approval prediction,
local SHAP explanation, bounded official-policy retrieval, and complete
grounded application analysis. The first three reuse the persisted CatBoost
champion, `explicar_solicitud()` and `retrieve_policy_context()` respectively.
The fourth invokes the existing deterministic LangGraph as a whole; MCP does
not implement a second orchestration path or an autonomous agent.

The application schema contains exactly the 16 approved raw V2 predictors and
rejects arbitrary fields. `LoanApproved` and the four `audit_only` attributes
are absent from the tool schemas and rejected as extras. Retrieval constrains
`top_k` to 1--10. All outputs have Pydantic/JSON schemas and contain only
JSON-serializable domain data, controlled warnings/errors and public source
metadata.

One lifespan-scoped resource manager lazily initializes and reuses the
champion, benchmark summary, local SentenceTransformer, embedded Qdrant client
and compiled LangGraph for the lifetime of a client session. It closes owned
resources on shutdown. API keys are left to the Gemini SDK environment and are
not placed in schemas, state, artifacts or responses.

The real integration test launches the module as a new `stdio` subprocess,
performs the MCP handshake, discovers exactly four tools, inspects their
schemas and executes prediction, XAI and retrieval. LangGraph/Gemini is covered
with a mocked provider in unit tests so the suite makes no live GenAI calls.
Validation evidence and examples are stored under `artifacts/mcp/`. No DagsHub
run is required because this phase changes the serving contract rather than ML
or retrieval behavior.

### Phase 9
End-to-end demo.

### Phase 10
Documentation, model card and GitHub release v2.0.0.

---

## 12. Definition of done for V2

Version 2 is considered complete when:

- `LoanApproved` remains the prediction target.
- The dataset has a documented dictionary of variables.
- Feature engineering is reproducible.
- Multiple ML models have been evaluated.
- Experiments are tracked with MLflow.
- A champion model has been selected using appropriate metrics.
- Model explanations are based on actual model evidence.
- RAG retrieves relevant credit-policy information.
- LangGraph orchestrates the ML and retrieval components.
- GenAI produces grounded analyst explanations.
- MCP exposes prediction, XAI, retrieval and full grounded analysis as typed
  local tools.
- The project can execute end-to-end.
- The README documents the architecture and results.
- The model card reflects the final model.
- Git history includes structured development and a v2.0.0 release.
