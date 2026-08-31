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
split. Cross-validation and future model selection operate only on the 80%
training partition. Feature engineering, median/mode imputation and
`OneHotEncoder(handle_unknown="ignore")` are composed inside the sklearn
pipeline so every future CV fold remains leakage-safe. No V2 estimator has
been selected or trained in this phase.

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
Model explainability.

### Phase 6
Credit-policy RAG with Qdrant.

### Phase 7
LangGraph agent with ML and RAG tools.

### Phase 8
FastMCP integration.

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
- The project can execute end-to-end.
- The README documents the architecture and results.
- The model card reflects the final model.
- Git history includes structured development and a v2.0.0 release.
