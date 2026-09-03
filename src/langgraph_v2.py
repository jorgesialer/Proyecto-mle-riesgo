"""Orquestacion determinista ML -> XAI -> RAG -> Gemini para V2.

LangGraph coordina componentes existentes; no toma decisiones autonomas ni
reemplaza al champion de Machine Learning. Gemini solo sintetiza evidencia que
supero guardrails explicitos de procedencia y completitud.
"""

from __future__ import annotations

import argparse
import json
import operator
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Callable, Mapping, Sequence, TypedDict
from urllib.parse import urlparse

import mlflow
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient

from src.construir_dataset_v2 import (
    AUDIT_ONLY_COLUMNS,
    MODEL_PREDICTOR_COLUMNS,
    TARGET_COLUMN,
)
from src.entrenamiento_v2 import EXPERIMENT_NAME
from src.mlflow_utils import TrackingConfig, configurar_mlflow, validar_run_name
from src.preprocesamiento_v2 import validar_columnas_predictoras_v2
from src.rag_v2 import (
    COLLECTION_NAME,
    DEFAULT_QDRANT_PATH,
    OFFICIAL_HOSTS,
    LocalSentenceEmbedder,
    construir_queries_desde_evidence,
    retrieve_policy_context,
)
from src.xai_v2 import (
    DEFAULT_CHAMPION_PATH,
    DEFAULT_DATASET_PATH,
    DEFAULT_SUMMARY_PATH,
    cargar_champion,
    cargar_resumen,
    explicar_solicitud,
    localizar_clase_positiva,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "langgraph"
DEFAULT_GENAI_PROVIDER = "gemini"
DEFAULT_GENAI_MODEL = "gemini-3.5-flash-lite"
PROMPT_VERSION = "grounded_generation_v1"
GUARDRAIL_VERSION = "official_sources_and_xai_coverage_v1"
RUN_NAME = "langgraph_grounded_generation_v1"
RUN_NAME_V2 = "langgraph_grounded_generation_v2"
MAX_GENERATION_ATTEMPTS = 3
RETRY_INITIAL_DELAY_SECONDS = 1.0
NODE_ORDER = (
    "validate_input",
    "predict_ml",
    "explain_with_shap",
    "build_rag_queries",
    "retrieve_policy_context",
    "evidence_guardrail",
    "generate_grounded_response",
)
REQUIRED_RESULT_METADATA = (
    "text",
    "source_id",
    "source_title",
    "institution",
    "source_url",
    "page_or_section",
    "chunk_id",
    "score",
)
CITATION_ID_PATTERN = re.compile(r"^\[?\s*[Ss](\d+)\s*\]?$")
CITATION_IN_TEXT_PATTERN = re.compile(
    r"\[\s*[Ss]\d+\s*\]|(?<![\w\[])[Ss]\d+(?![\w\]])"
)
CANONICAL_CITATION_PATTERN = re.compile(r"\[S\d+\]")


class WorkflowState(TypedDict, total=False):
    raw_application: dict[str, Any]
    validated_application: dict[str, Any]
    prediction: int
    prediction_label: str
    probability: float
    xai_evidence: dict[str, Any]
    rag_queries: list[dict[str, str]]
    policy_evidence: list[dict[str, Any]]
    evidence_sufficiency: dict[str, Any]
    insufficient_evidence: bool
    final_response: str
    structured_response: dict[str, Any]
    sources: list[dict[str, Any]]
    gemini_called: bool
    generation_attempts: int
    generation_errors: list[dict[str, Any]]
    generation_status: str
    provider_status: str
    citation_normalization_count: int
    node_trace: Annotated[list[str], operator.add]
    warnings: Annotated[list[str], operator.add]
    errors: Annotated[list[str], operator.add]


class GeminiGroundedContent(BaseModel):
    model_explanation: str = Field(
        description="Non-causal summary of the supplied ML/SHAP evidence"
    )
    policy_context: str = Field(
        description="Policy context supported only by supplied citation identifiers"
    )
    final_assessment: str = Field(
        description="Careful synthesis that separates prediction from policy context"
    )
    cited_source_ids: list[str] = Field(
        description="Citation identifiers such as [S1], limited to supplied evidence"
    )


@dataclass(frozen=True)
class GenAIConfig:
    provider: str
    model: str


def cargar_config_genai(env: Mapping[str, str] | None = None) -> GenAIConfig:
    """Resuelve provider/model sin leer ni propagar credenciales."""
    values = os.environ if env is None else env
    provider = values.get("GENAI_PROVIDER", DEFAULT_GENAI_PROVIDER).strip().lower()
    model = values.get("GENAI_MODEL", DEFAULT_GENAI_MODEL).strip()
    if provider != "gemini":
        raise ValueError(
            f"GENAI_PROVIDER no soportado en esta fase: {provider!r}; use 'gemini'"
        )
    if not model:
        raise ValueError("GENAI_MODEL no puede estar vacio")
    return GenAIConfig(provider=provider, model=model)


@dataclass
class WorkflowResources:
    pipeline: Any
    summary: dict[str, Any]
    embedder: Any
    qdrant_client: Any
    genai_config: GenAIConfig
    gemini_client: Any | None = None
    gemini_client_factory: Callable[[], Any] | None = None
    xai_top_n: int = 5
    rag_top_k: int = 3
    owns_qdrant_client: bool = False
    owns_gemini_client: bool = False
    generation_max_attempts: int = MAX_GENERATION_ATTEMPTS
    retry_initial_delay_seconds: float = RETRY_INITIAL_DELAY_SECONDS
    sleep_fn: Callable[[float], None] = time.sleep


def _json_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return _json_value(value)


def _redact_secrets(value: Any) -> str:
    text = str(value)
    for name in ("GOOGLE_API_KEY", "GEMINI_API_KEY"):
        secret = os.environ.get(name)
        if secret:
            text = text.replace(secret, "[REDACTED]")
    return text


def _normalize_citation_id(raw: str) -> str:
    """Normaliza solo IDs S+digitos; cualquier otro string queda intacto."""
    match = CITATION_ID_PATTERN.fullmatch(raw)
    return f"[S{match.group(1)}]" if match else raw


def _normalize_citations_in_text(text: str) -> tuple[str, int]:
    count = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal count
        raw = match.group(0)
        normalized = _normalize_citation_id(raw)
        if normalized != raw:
            count += 1
        return normalized

    return CITATION_IN_TEXT_PATTERN.sub(replace, text), count


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def construir_prompt_grounded(
    model_evidence: dict[str, Any],
    policy_evidence: Sequence[dict[str, Any]],
    warnings: Sequence[str],
) -> str:
    """Construye un prompt versionado con evidencia separada y reglas fail-closed."""
    instructions = """You are preparing a Spanish-language analytical explanation.
Follow every rule below:
- Do not claim causality.
- Do not claim that any retrieved policy caused the historical model prediction.
- Do not invent rules, thresholds, facts, or sources.
- SHAP values are contributions in raw margin/log-odds space, not direct changes in probability.
- Clearly distinguish the ML prediction and model evidence from normative policy context.
- Cite only citation identifiers present in POLICY_EVIDENCE, using their exact [S#] form.
- Use citation IDs exactly in the form [S#]. Correct: [S3]. Incorrect: S3.
- If evidence is missing or ambiguous, state that limitation explicitly.
- Do not turn this output into an autonomous approval or denial decision.
"""
    return (
        f"PROMPT_VERSION: {PROMPT_VERSION}\n\n"
        f"INSTRUCTIONS:\n{instructions}\n"
        "MODEL_EVIDENCE:\n"
        + json.dumps(_json_safe(model_evidence), ensure_ascii=False, allow_nan=False)
        + "\n\nPOLICY_EVIDENCE:\n"
        + json.dumps(_json_safe(list(policy_evidence)), ensure_ascii=False, allow_nan=False)
        + "\n\nWARNINGS:\n"
        + json.dumps(list(warnings), ensure_ascii=False, allow_nan=False)
    )


def _render_final_text(output: dict[str, Any]) -> str:
    return (
        f"Prediccion del modelo: {output['prediction']} "
        f"(probabilidad={output['probability']:.6f})\n\n"
        f"Evidencia del modelo:\n{output['model_explanation']}\n\n"
        f"Contexto normativo:\n{output['policy_context']}\n\n"
        f"Evaluacion final:\n{output['final_assessment']}"
    )


def _abstention_output(
    state: WorkflowState,
    reason: str,
    additional_warnings: Sequence[str] = (),
    *,
    model_explanation: str | None = None,
    policy_context: str | None = None,
) -> tuple[dict[str, Any], str]:
    warnings = list(state.get("warnings", [])) + list(additional_warnings)
    errors = list(state.get("errors", []))
    if errors:
        warnings.extend(f"Error de workflow: {message}" for message in errors)
    output = {
        "prediction": state.get("prediction_label", "unavailable"),
        "probability": float(state.get("probability", 0.0)),
        "model_explanation": model_explanation
        or (
            "La evidencia del modelo no puede sintetizarse de forma grounded "
            "en esta ejecucion."
        ),
        "policy_context": policy_context
        or "Evidencia documental oficial insuficiente o invalida.",
        "final_assessment": f"Abstencion controlada: {reason}",
        "sources": [],
        "warnings": list(dict.fromkeys(warnings)),
    }
    return output, _render_final_text(output)


class GroundedCreditWorkflow:
    """Workflow lineal compilado; todas las dependencias se reutilizan por run."""

    def __init__(self, resources: WorkflowResources) -> None:
        self.resources = resources
        self.graph = self._build_graph()

    @classmethod
    def from_artifacts(
        cls,
        *,
        champion_path: Path = DEFAULT_CHAMPION_PATH,
        summary_path: Path = DEFAULT_SUMMARY_PATH,
        gemini_client: Any | None = None,
        genai_config: GenAIConfig | None = None,
        rag_top_k: int = 3,
    ) -> "GroundedCreditWorkflow":
        pipeline = cargar_champion(champion_path)
        summary = cargar_resumen(summary_path)
        embedder = LocalSentenceEmbedder()
        qdrant_client = QdrantClient(path=str(DEFAULT_QDRANT_PATH))
        resources = WorkflowResources(
            pipeline=pipeline,
            summary=summary,
            embedder=embedder,
            qdrant_client=qdrant_client,
            genai_config=genai_config or cargar_config_genai(),
            gemini_client=gemini_client,
            gemini_client_factory=lambda: genai.Client(
                http_options=types.HttpOptions(
                    retry_options=types.HttpRetryOptions(attempts=1)
                )
            ),
            rag_top_k=rag_top_k,
            owns_qdrant_client=True,
            owns_gemini_client=False,
        )
        return cls(resources)

    def _build_graph(self):
        builder = StateGraph(WorkflowState)
        builder.add_node("validate_input", self._validate_input)
        builder.add_node("predict_ml", self._predict_ml)
        builder.add_node("explain_with_shap", self._explain_with_shap)
        builder.add_node("build_rag_queries", self._build_rag_queries)
        builder.add_node("retrieve_policy_context", self._retrieve_policy_context)
        builder.add_node("evidence_guardrail", self._evidence_guardrail)
        builder.add_node(
            "generate_grounded_response", self._generate_grounded_response
        )
        builder.add_edge(START, NODE_ORDER[0])
        for current, following in zip(NODE_ORDER, NODE_ORDER[1:]):
            builder.add_edge(current, following)
        builder.add_edge(NODE_ORDER[-1], END)
        return builder.compile()

    def invoke(self, raw_application: dict[str, Any]) -> WorkflowState:
        initial: WorkflowState = {
            "raw_application": _json_safe(dict(raw_application)),
            "warnings": [],
            "errors": [],
            "node_trace": [],
            "gemini_called": False,
            "generation_attempts": 0,
            "generation_errors": [],
            "generation_status": "not_started",
            "provider_status": "not_called",
            "citation_normalization_count": 0,
        }
        result = self.graph.invoke(initial)
        json.dumps(_json_safe(result["structured_response"]), allow_nan=False)
        return result

    def close(self) -> None:
        if self.resources.owns_qdrant_client:
            self.resources.qdrant_client.close()
            self.resources.owns_qdrant_client = False
        if self.resources.owns_gemini_client and self.resources.gemini_client:
            self.resources.gemini_client.close()
            self.resources.owns_gemini_client = False

    def _validate_input(self, state: WorkflowState) -> dict[str, Any]:
        raw = state.get("raw_application")
        errors = []
        if not isinstance(raw, dict):
            errors.append("raw_application debe ser un objeto JSON")
        else:
            forbidden = (set(AUDIT_ONLY_COLUMNS) | {TARGET_COLUMN}) & set(raw)
            if forbidden:
                errors.append(
                    "atributos audit-only/target prohibidos: " + str(sorted(forbidden))
                )
            try:
                validar_columnas_predictoras_v2(raw.keys())
            except ValueError as exc:
                errors.append(str(exc))
        update: dict[str, Any] = {
            "node_trace": ["validate_input"],
            "errors": errors,
        }
        if not errors and raw is not None:
            update["validated_application"] = {
                column: raw[column] for column in MODEL_PREDICTOR_COLUMNS
            }
        return update

    def _predict_ml(self, state: WorkflowState) -> dict[str, Any]:
        update: dict[str, Any] = {"node_trace": ["predict_ml"]}
        if state.get("errors") or "validated_application" not in state:
            return update
        raw = pd.DataFrame(
            [state["validated_application"]], columns=MODEL_PREDICTOR_COLUMNS
        )
        estimator = self.resources.pipeline.named_steps["estimador"]
        positive_index = localizar_clase_positiva(estimator)
        prediction = int(self.resources.pipeline.predict(raw)[0])
        probability = float(
            self.resources.pipeline.predict_proba(raw)[0, positive_index]
        )
        update |= {
            "prediction": prediction,
            "prediction_label": "approved" if prediction == 1 else "denied",
            "probability": probability,
        }
        return update

    def _explain_with_shap(self, state: WorkflowState) -> dict[str, Any]:
        update: dict[str, Any] = {"node_trace": ["explain_with_shap"]}
        if state.get("errors") or "validated_application" not in state:
            return update
        evidence = explicar_solicitud(
            self.resources.pipeline,
            state["validated_application"],
            top_n=self.resources.xai_top_n,
            summary=self.resources.summary,
        )
        if evidence["prediction"] != state.get("prediction") or not np.isclose(
            evidence["probability"], state.get("probability", np.nan)
        ):
            return update | {
                "errors": ["Prediccion ML y evidence package XAI no coinciden"]
            }
        return update | {"xai_evidence": evidence}

    def _build_rag_queries(self, state: WorkflowState) -> dict[str, Any]:
        update: dict[str, Any] = {"node_trace": ["build_rag_queries"]}
        if state.get("errors") or "xai_evidence" not in state:
            return update
        payload = construir_queries_desde_evidence(state["xai_evidence"])
        return update | {"rag_queries": payload["queries"]}

    def _retrieve_policy_context(self, state: WorkflowState) -> dict[str, Any]:
        update: dict[str, Any] = {"node_trace": ["retrieve_policy_context"]}
        if state.get("errors"):
            return update
        evidence = []
        errors = []
        for query in state.get("rag_queries", []):
            try:
                retrieval = retrieve_policy_context(
                    query["query"],
                    k=self.resources.rag_top_k,
                    collection_name=COLLECTION_NAME,
                    embedder=self.resources.embedder,
                    client=self.resources.qdrant_client,
                )
            except Exception as exc:  # retrieval debe convertirse en abstencion
                errors.append(f"Retrieval fallo para {query['origin_feature']}: {exc}")
                continue
            evidence.append(
                {
                    "origin_feature": query["origin_feature"],
                    "query": retrieval["query"],
                    "results": retrieval["results"],
                }
            )
        return update | {"policy_evidence": evidence, "errors": errors}

    def _evidence_guardrail(self, state: WorkflowState) -> dict[str, Any]:
        reasons: list[str] = []
        warnings = [
            "Retrieval scores are ranking signals, not calibrated confidence.",
            "No numeric rejection threshold is configured.",
        ]
        if state.get("errors"):
            reasons.append("El workflow contiene errores previos")
        queries = state.get("rag_queries", [])
        evidence = state.get("policy_evidence", [])
        if not queries:
            reasons.append("No se generaron queries desde factores XAI")
        if not evidence or not any(item.get("results") for item in evidence):
            reasons.append("Retrieval no devolvio resultados")

        xai_factors = []
        xai = state.get("xai_evidence", {})
        xai_factors.extend(xai.get("top_positive_factors") or [])
        xai_factors.extend(xai.get("top_negative_factors") or [])
        relevant_features = {
            str(factor.get("model_feature")) for factor in xai_factors
        }
        covered_features: set[str] = set()
        sources: list[dict[str, Any]] = []
        citation_by_chunk: dict[str, str] = {}
        guarded_evidence: list[dict[str, Any]] = []

        for item in evidence:
            guarded_results = []
            for result in item.get("results", []):
                missing = [
                    field
                    for field in REQUIRED_RESULT_METADATA
                    if field not in result
                    or result[field] is None
                    or (isinstance(result[field], str) and not result[field].strip())
                ]
                if missing:
                    reasons.append(
                        f"Metadata incompleta en result {result.get('chunk_id')}: {missing}"
                    )
                    continue
                parsed = urlparse(str(result["source_url"]))
                if parsed.scheme != "https" or parsed.hostname not in OFFICIAL_HOSTS:
                    reasons.append(
                        f"Fuente fuera de allowlist: {result['source_url']}"
                    )
                    continue
                chunk_id = str(result["chunk_id"])
                if chunk_id not in citation_by_chunk:
                    citation_id = f"[S{len(citation_by_chunk) + 1}]"
                    citation_by_chunk[chunk_id] = citation_id
                    sources.append(
                        {
                            "citation_id": citation_id,
                            "source_id": str(result["source_id"]),
                            "source_title": str(result["source_title"]),
                            "institution": str(result["institution"]),
                            "source_url": str(result["source_url"]),
                            "page_or_section": str(result["page_or_section"]),
                            "chunk_id": chunk_id,
                        }
                    )
                guarded_results.append(
                    dict(result, citation_id=citation_by_chunk[chunk_id])
                )
            if guarded_results:
                origin = str(item.get("origin_feature", ""))
                covered_features.add(origin)
                guarded_evidence.append(
                    {
                        "origin_feature": origin,
                        "query": item.get("query"),
                        "results": guarded_results,
                    }
                )

        relevant_coverage = sorted(relevant_features & covered_features)
        if not relevant_coverage:
            reasons.append(
                "No hay evidencia documental para ningun factor XAI relevante"
            )
        insufficient = bool(reasons)
        status = {
            "insufficient_evidence": insufficient,
            "guardrail_version": GUARDRAIL_VERSION,
            "reasons": list(dict.fromkeys(reasons)),
            "official_source_allowlist": sorted(OFFICIAL_HOSTS),
            "xai_relevant_features": sorted(relevant_features),
            "covered_xai_features": relevant_coverage,
            "retrieved_source_count": len(sources),
            "numeric_score_threshold": None,
        }
        return {
            "node_trace": ["evidence_guardrail"],
            "policy_evidence": guarded_evidence,
            "sources": sources,
            "evidence_sufficiency": status,
            "insufficient_evidence": insufficient,
            "warnings": warnings,
        }

    def _get_gemini_client(self) -> Any:
        if self.resources.gemini_client is None:
            factory = self.resources.gemini_client_factory or (
                lambda: genai.Client(
                    http_options=types.HttpOptions(
                        retry_options=types.HttpRetryOptions(attempts=1)
                    )
                )
            )
            self.resources.gemini_client = factory()
            self.resources.owns_gemini_client = True
        return self.resources.gemini_client

    @staticmethod
    def _is_retryable_generation_error(exc: Exception) -> bool:
        if isinstance(exc, json.JSONDecodeError):
            return True
        if isinstance(exc, genai_errors.UnknownApiResponseError):
            return True
        if isinstance(exc, genai_errors.ServerError):
            code = getattr(exc, "code", None)
            status = str(getattr(exc, "status", "")).upper()
            return code == 503 or status == "UNAVAILABLE"
        return False

    @staticmethod
    def _generation_error_record(
        attempt: int,
        exc: Exception,
        *,
        retryable: bool,
    ) -> dict[str, Any]:
        return {
            "attempt": attempt,
            "error_type": type(exc).__name__,
            "message": _redact_secrets(exc),
            "provider_code": getattr(exc, "code", None),
            "provider_status": getattr(exc, "status", None),
            "retryable": retryable,
        }

    def _call_gemini_with_retries(
        self,
        prompt: str,
    ) -> tuple[GeminiGroundedContent | None, int, list[dict[str, Any]], str]:
        errors: list[dict[str, Any]] = []
        max_attempts = self.resources.generation_max_attempts
        if max_attempts < 1 or max_attempts > MAX_GENERATION_ATTEMPTS:
            raise ValueError(
                f"generation_max_attempts debe estar entre 1 y {MAX_GENERATION_ATTEMPTS}"
            )
        for attempt in range(1, max_attempts + 1):
            try:
                response = self._get_gemini_client().models.generate_content(
                    model=self.resources.genai_config.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0,
                        max_output_tokens=2_048,
                        response_mime_type="application/json",
                        response_json_schema=GeminiGroundedContent.model_json_schema(),
                    ),
                )
                parsed = GeminiGroundedContent.model_validate(
                    response.parsed
                    if getattr(response, "parsed", None) is not None
                    else json.loads(response.text)
                )
                return parsed, attempt, errors, "success"
            except Exception as exc:
                retryable = self._is_retryable_generation_error(exc)
                errors.append(
                    self._generation_error_record(
                        attempt,
                        exc,
                        retryable=retryable,
                    )
                )
                if not retryable:
                    return None, attempt, errors, "permanent_error"
                if attempt < max_attempts:
                    delay = self.resources.retry_initial_delay_seconds * (
                        2 ** (attempt - 1)
                    )
                    self.resources.sleep_fn(delay)
        return None, max_attempts, errors, "unavailable"

    def _generate_grounded_response(self, state: WorkflowState) -> dict[str, Any]:
        trace = {"node_trace": ["generate_grounded_response"]}
        if state.get("insufficient_evidence", True):
            reasons = state.get("evidence_sufficiency", {}).get("reasons", [])
            reason = "; ".join(reasons) or "evidencia insuficiente"
            output, text = _abstention_output(state, reason)
            return trace | {
                "structured_response": output,
                "final_response": text,
                "gemini_called": False,
                "generation_attempts": 0,
                "generation_errors": [],
                "generation_status": "abstained_insufficient_evidence",
                "provider_status": "not_called",
                "citation_normalization_count": 0,
            }

        prompt = construir_prompt_grounded(
            state["xai_evidence"],
            state["policy_evidence"],
            state.get("warnings", []),
        )
        parsed, attempts, attempt_errors, provider_status = (
            self._call_gemini_with_retries(prompt)
        )
        if parsed is None:
            last_error = attempt_errors[-1] if attempt_errors else {}
            warning = (
                "Gemini generation failed after "
                f"{attempts} attempt(s): {last_error.get('error_type', 'unknown')}: "
                f"{last_error.get('message', 'unknown error')}"
            )
            generation_status = (
                "provider_unavailable"
                if provider_status == "unavailable"
                else "provider_error"
            )
            output, text = _abstention_output(
                state,
                "fallo la generacion grounded",
                [warning],
                model_explanation=(
                    "La prediccion y la evidencia XAI se generaron, pero no se "
                    "produjo una sintesis grounded."
                ),
                policy_context=(
                    "El contexto oficial fue recuperado y validado, pero no pudo "
                    "sintetizarse porque el proveedor GenAI no estuvo disponible."
                ),
            )
            return trace | {
                "structured_response": output,
                "final_response": text,
                "gemini_called": True,
                "generation_attempts": attempts,
                "generation_errors": attempt_errors,
                "generation_status": generation_status,
                "provider_status": provider_status,
                "citation_normalization_count": 0,
                "warnings": [warning],
                "errors": [warning],
            }

        allowed_sources = {
            source["citation_id"]: source for source in state.get("sources", [])
        }
        normalized_fields: dict[str, str] = {}
        normalization_count = 0
        for field in ("model_explanation", "policy_context", "final_assessment"):
            normalized_text, field_count = _normalize_citations_in_text(
                getattr(parsed, field)
            )
            normalized_fields[field] = normalized_text
            normalization_count += field_count
        normalized_declared = []
        for raw_citation in parsed.cited_source_ids:
            normalized = _normalize_citation_id(raw_citation)
            normalization_count += int(normalized != raw_citation)
            normalized_declared.append(normalized)

        generated_text = " ".join(
            [
                normalized_fields["model_explanation"],
                normalized_fields["policy_context"],
                normalized_fields["final_assessment"],
            ]
        )
        mentioned = set(CANONICAL_CITATION_PATTERN.findall(generated_text))
        declared = set(normalized_declared)
        invalid = sorted((mentioned | declared) - set(allowed_sources))
        valid_citations = sorted(
            (mentioned | declared) & set(allowed_sources),
            key=lambda value: int(value[2:-1]),
        )
        if invalid or not valid_citations:
            reason = (
                f"Gemini produjo citations no permitidas: {invalid}"
                if invalid
                else "Gemini no cito evidencia recuperada"
            )
            output, text = _abstention_output(state, reason, [reason])
            return trace | {
                "structured_response": output,
                "final_response": text,
                "gemini_called": True,
                "generation_attempts": attempts,
                "generation_errors": attempt_errors,
                "generation_status": "output_validation_failed",
                "provider_status": provider_status,
                "citation_normalization_count": normalization_count,
                "warnings": [reason],
                "errors": [reason],
            }

        output = {
            "prediction": state["prediction_label"],
            "probability": float(state["probability"]),
            "model_explanation": normalized_fields["model_explanation"],
            "policy_context": normalized_fields["policy_context"],
            "final_assessment": normalized_fields["final_assessment"],
            "sources": [allowed_sources[value] for value in valid_citations],
            "warnings": list(dict.fromkeys(state.get("warnings", []))),
        }
        json.dumps(_json_safe(output), allow_nan=False)
        return trace | {
            "structured_response": output,
            "final_response": _render_final_text(output),
            "gemini_called": True,
            "generation_attempts": attempts,
            "generation_errors": attempt_errors,
            "generation_status": "success",
            "provider_status": provider_status,
            "citation_normalization_count": normalization_count,
        }


def registrar_run_langgraph(
    tracking: TrackingConfig,
    output_dir: Path,
    state: WorkflowState,
    *,
    run_name: str = RUN_NAME,
    genai_config: GenAIConfig,
    rag_top_k: int = 3,
) -> dict[str, Any]:
    validar_run_name(run_name)
    with mlflow.start_run(experiment_id=tracking.experiment_id, run_name=run_name) as run:
        mlflow.set_tags(
            {
                "stage": "grounded_generation",
                "orchestrator": "langgraph_deterministic",
                "generation_status": state.get("generation_status", "unknown"),
                "provider_status": state.get("provider_status", "unknown"),
                "mcp_implemented": "false",
                "docker_required": "false",
            }
        )
        mlflow.log_params(
            {
                "node_order": " -> ".join(NODE_ORDER),
                "node_count": len(NODE_ORDER),
                "genai_provider": genai_config.provider,
                "genai_model": genai_config.model,
                "prompt_version": PROMPT_VERSION,
                "guardrail_version": GUARDRAIL_VERSION,
                "retrieval_top_k": rag_top_k,
                "numeric_score_threshold": "none",
                "generation_max_attempts": MAX_GENERATION_ATTEMPTS,
                "retry_initial_delay_seconds": RETRY_INITIAL_DELAY_SECONDS,
                "retry_backoff_multiplier": 2,
            }
        )
        mlflow.log_metrics(
            {
                "retrieved_source_count": float(len(state.get("sources", []))),
                "insufficient_evidence": float(
                    bool(state.get("insufficient_evidence", True))
                ),
                "gemini_called": float(bool(state.get("gemini_called", False))),
                "generation_attempts": float(state.get("generation_attempts", 0)),
                "generation_error_count": float(
                    len(state.get("generation_errors", []))
                ),
                "citation_normalization_count": float(
                    state.get("citation_normalization_count", 0)
                ),
            }
        )
        run_metadata = {
            "run_id": run.info.run_id,
            "run_name": run_name,
            "backend": tracking.backend,
            "experiment_name": tracking.experiment_name,
            "tracking_ui": tracking.ui_url,
            "genai_provider": genai_config.provider,
            "genai_model": genai_config.model,
            "citation_normalization_count": state.get(
                "citation_normalization_count", 0
            ),
        }
        _write_json(output_dir / "run_metadata.json", run_metadata)
        mlflow.log_artifacts(str(output_dir), artifact_path="langgraph")
        return run_metadata


def ejecutar_integracion(
    *,
    backend: str | None = None,
    log_mlflow: bool = True,
    dataset_path: Path = DEFAULT_DATASET_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    load_dotenv(PROJECT_ROOT / ".env")
    genai_config = cargar_config_genai()
    data = pd.read_csv(dataset_path, nrows=1)
    application = {
        column: _json_value(data.iloc[0][column])
        for column in MODEL_PREDICTOR_COLUMNS
    }
    workflow = GroundedCreditWorkflow.from_artifacts(genai_config=genai_config)
    try:
        state = workflow.invoke(application)
    finally:
        workflow.close()

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "input_application.json", application)
    _write_json(
        output_dir / "generation_metadata.json",
        {
            "generation_attempts": state.get("generation_attempts", 0),
            "generation_errors": state.get("generation_errors", []),
            "generation_status": state.get("generation_status"),
            "provider_status": state.get("provider_status"),
            "max_attempts": workflow.resources.generation_max_attempts,
            "initial_delay_seconds": workflow.resources.retry_initial_delay_seconds,
            "backoff_multiplier": 2,
            "genai_provider": genai_config.provider,
            "genai_model": genai_config.model,
            "citation_normalization_count": state.get(
                "citation_normalization_count", 0
            ),
        },
    )
    if state.get("generation_status") == "success":
        _write_json(
            output_dir / "grounded_success_example.json",
            state["structured_response"],
        )
    else:
        _write_json(
            output_dir / "grounded_example.json",
            state["structured_response"],
        )
    _write_json(output_dir / "source_metadata.json", state.get("sources", []))
    _write_json(
        output_dir / "graph_config.json",
        {
            "node_order": list(NODE_ORDER),
            "state_schema": sorted(WorkflowState.__annotations__),
            "genai_provider": genai_config.provider,
            "genai_model": genai_config.model,
            "prompt_version": PROMPT_VERSION,
            "guardrail_version": GUARDRAIL_VERSION,
            "retrieval_top_k": workflow.resources.rag_top_k,
            "numeric_score_threshold": None,
            "generation_max_attempts": workflow.resources.generation_max_attempts,
            "retry_initial_delay_seconds": workflow.resources.retry_initial_delay_seconds,
            "retry_backoff_multiplier": 2,
        },
    )
    (output_dir / "prompt_template.txt").write_text(
        construir_prompt_grounded(
            {"<model_evidence>": "supplied at runtime"},
            [{"<policy_evidence>": "supplied at runtime"}],
            ["<warnings supplied at runtime>"],
        )
        + "\n",
        encoding="utf-8",
    )

    run_metadata = None
    if log_mlflow and state.get("generation_status") == "success":
        tracking = configurar_mlflow(
            PROJECT_ROOT,
            backend=backend,
            experiment_name=EXPERIMENT_NAME,
        )
        run_metadata = registrar_run_langgraph(
            tracking,
            output_dir,
            state,
            run_name=RUN_NAME_V2,
            genai_config=genai_config,
            rag_top_k=workflow.resources.rag_top_k,
        )
    return {
        "response": state["structured_response"],
        "final_text": state["final_response"],
        "node_trace": state["node_trace"],
        "guardrail": state["evidence_sufficiency"],
        "generation_attempts": state.get("generation_attempts", 0),
        "generation_errors": state.get("generation_errors", []),
        "generation_status": state.get("generation_status"),
        "provider_status": state.get("provider_status"),
        "citation_normalization_count": state.get(
            "citation_normalization_count", 0
        ),
        "genai_provider": genai_config.provider,
        "genai_model": genai_config.model,
        "mlflow": run_metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ejecuta LangGraph V2 con ML, SHAP, RAG y Gemini grounded"
    )
    parser.add_argument("--backend", choices=("local", "dagshub"), default=None)
    parser.add_argument("--skip-mlflow", action="store_true")
    args = parser.parse_args()
    result = ejecutar_integracion(
        backend=args.backend,
        log_mlflow=not args.skip_mlflow,
    )
    print(json.dumps(_json_safe(result), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
