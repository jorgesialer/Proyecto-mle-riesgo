"""Servidor MCP V2 para exponer ML, XAI, RAG y LangGraph existentes.

El transporte por defecto es stdio. Los recursos pesados se crean de forma
perezosa dentro del lifespan del servidor y se reutilizan durante la sesion.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import version as package_version
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Literal

import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types as genai_types
from mcp.client import Client
from mcp.server.mcpserver import Context, MCPServer
from mcp.types import ToolAnnotations
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, JsonValue
from qdrant_client import QdrantClient

from src.construir_dataset_v2 import (
    AUDIT_ONLY_COLUMNS,
    MODEL_PREDICTOR_COLUMNS,
    TARGET_COLUMN,
)
from src.langgraph_v2 import (
    GroundedCreditWorkflow,
    WorkflowResources,
    cargar_config_genai,
)
from src.preprocesamiento_v2 import validar_columnas_predictoras_v2
from src.rag_v2 import (
    COLLECTION_NAME,
    DEFAULT_QDRANT_PATH,
    LocalSentenceEmbedder,
    retrieve_policy_context,
)
from src.xai_v2 import (
    DEFAULT_CHAMPION_PATH,
    DEFAULT_SUMMARY_PATH,
    cargar_champion,
    cargar_resumen,
    explicar_solicitud,
    localizar_clase_positiva,
)


SERVER_NAME = "credit-approval-v2"
SERVER_VERSION = "1.0.0"
TRANSPORT = "stdio"
TOOL_NAMES = (
    "predict_credit_application",
    "explain_credit_application",
    "retrieve_credit_policy",
    "analyze_credit_application",
)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MCP_ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "mcp"
DEFAULT_DATASET_PATH = PROJECT_ROOT / "data" / "hmda_2023_loan_approval_v2.csv"
DEFAULT_GROUNDED_FIXTURE = (
    PROJECT_ROOT / "artifacts" / "langgraph" / "grounded_success_example.json"
)


class StrictModel(BaseModel):
    """Base de contrato: no admite argumentos o campos no declarados."""

    model_config = ConfigDict(extra="forbid")


class CreditApplication(StrictModel):
    """Las 16 variables crudas autorizadas por el contrato HMDA V2."""

    income: FiniteFloat | None
    loan_amount: FiniteFloat | None
    loan_term: FiniteFloat | None
    loan_purpose: int | str | None
    loan_type: int | str | None
    lien_status: int | str | None
    preapproval: int | str | None
    debt_to_income_ratio: str | FiniteFloat | None
    combined_loan_to_value_ratio: FiniteFloat | None
    property_value: FiniteFloat | None
    occupancy_type: int | str | None
    construction_method: int | str | None
    total_units: int | str | None
    submission_of_application: int | str | None
    interest_only_payment: int | str | None
    balloon_payment: int | str | None

    def as_model_input(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json")
        validar_columnas_predictoras_v2(payload.keys())
        return {column: payload[column] for column in MODEL_PREDICTOR_COLUMNS}


class ModelIdentity(StrictModel):
    name: str
    version: str | None
    selected_configuration: str | None
    dataset_version: str | None


class PredictionOutput(StrictModel):
    prediction: Literal[0, 1]
    prediction_label: Literal["approved", "denied"]
    probability: float = Field(ge=0.0, le=1.0)
    model: ModelIdentity
    warnings: list[str]


class ExplanationOutput(StrictModel):
    prediction: Literal[0, 1]
    prediction_label: Literal["approved", "denied"]
    probability: float = Field(ge=0.0, le=1.0)
    shap_output_space: str
    top_positive_factors: list[dict[str, JsonValue]]
    top_negative_factors: list[dict[str, JsonValue]]
    model: dict[str, JsonValue]
    dataset: dict[str, JsonValue]
    non_causality_warning: str
    evidence_package: dict[str, JsonValue]


class PolicyResult(StrictModel):
    text: str
    score: float
    source_id: str
    source_title: str
    institution: str
    source_url: str
    page_or_section: str
    chunk_id: str


class PolicyRetrievalOutput(StrictModel):
    query: str
    top_k: int
    results: list[PolicyResult]
    warnings: list[str]


class GroundedSource(StrictModel):
    citation_id: str
    source_id: str
    source_title: str
    institution: str
    source_url: str
    page_or_section: str
    chunk_id: str


class GroundedResponse(StrictModel):
    prediction: str
    probability: float
    model_explanation: str
    policy_context: str
    final_assessment: str
    sources: list[GroundedSource]
    warnings: list[str]


class AnalysisOutput(StrictModel):
    prediction: Literal[0, 1] | None
    prediction_label: Literal["approved", "denied"] | None
    probability: float | None
    xai_evidence: dict[str, JsonValue] | None
    rag_queries: list[dict[str, JsonValue]]
    policy_evidence: list[dict[str, JsonValue]]
    evidence_sufficiency: dict[str, JsonValue] | None
    grounded_response: GroundedResponse
    final_response: str
    sources: list[GroundedSource]
    warnings: list[str]
    errors: list[str]
    node_trace: list[str]
    gemini_called: bool
    generation_attempts: int
    generation_status: str
    provider_status: str
    citation_normalization_count: int


@dataclass
class MCPResourceManager:
    """Cache por lifespan; no carga modelos ni Qdrant al importar el modulo."""

    champion_path: Path = PROJECT_ROOT / DEFAULT_CHAMPION_PATH
    summary_path: Path = PROJECT_ROOT / DEFAULT_SUMMARY_PATH
    qdrant_path: Path = PROJECT_ROOT / DEFAULT_QDRANT_PATH
    gemini_client: Any | None = None
    load_environment: bool = True

    def __post_init__(self) -> None:
        self._lock = threading.RLock()
        self._analysis_lock = threading.Lock()
        self._pipeline: Any | None = None
        self._summary: dict[str, Any] | None = None
        self._embedder: Any | None = None
        self._qdrant_client: Any | None = None
        self._workflow: GroundedCreditWorkflow | None = None
        self.load_counts = {
            "pipeline": 0,
            "summary": 0,
            "embedder": 0,
            "qdrant_client": 0,
            "workflow": 0,
        }

    def model_resources(self) -> tuple[Any, dict[str, Any]]:
        with self._lock:
            if self._pipeline is None:
                self._pipeline = cargar_champion(self.champion_path)
                self.load_counts["pipeline"] += 1
            if self._summary is None:
                self._summary = cargar_resumen(self.summary_path)
                self.load_counts["summary"] += 1
            return self._pipeline, self._summary

    def rag_resources(self) -> tuple[Any, Any]:
        with self._lock:
            if self._embedder is None:
                self._embedder = LocalSentenceEmbedder()
                self.load_counts["embedder"] += 1
            if self._qdrant_client is None:
                self._qdrant_client = QdrantClient(path=str(self.qdrant_path))
                self.load_counts["qdrant_client"] += 1
            return self._embedder, self._qdrant_client

    def workflow(self) -> GroundedCreditWorkflow:
        with self._lock:
            if self._workflow is None:
                if self.load_environment:
                    load_dotenv(PROJECT_ROOT / ".env")
                pipeline, summary = self.model_resources()
                embedder, qdrant_client = self.rag_resources()
                supplied_client = self.gemini_client
                resources = WorkflowResources(
                    pipeline=pipeline,
                    summary=summary,
                    embedder=embedder,
                    qdrant_client=qdrant_client,
                    genai_config=cargar_config_genai(),
                    gemini_client=supplied_client,
                    gemini_client_factory=(
                        None
                        if supplied_client is not None
                        else lambda: genai.Client(
                            http_options=genai_types.HttpOptions(
                                retry_options=genai_types.HttpRetryOptions(attempts=1)
                            )
                        )
                    ),
                    owns_qdrant_client=False,
                    owns_gemini_client=False,
                )
                self._workflow = GroundedCreditWorkflow(resources)
                self.load_counts["workflow"] += 1
            return self._workflow

    def invoke_workflow(self, application: dict[str, Any]) -> dict[str, Any]:
        # El workflow comparte un cliente Gemini perezoso; se serializa su uso.
        with self._analysis_lock:
            return dict(self.workflow().invoke(application))

    def close(self) -> None:
        with self._lock:
            if self._workflow is not None:
                self._workflow.close()
            if self._qdrant_client is not None:
                self._qdrant_client.close()
                self._qdrant_client = None


@dataclass(frozen=True)
class MCPRuntime:
    resources: MCPResourceManager


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return value.name
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, float) and pd.isna(value):
        return None
    return value


def _controlled_error(operation: str, exc: Exception) -> ValueError:
    """Evita filtrar rutas, objetos internos o credenciales en errores MCP."""
    if isinstance(exc, ValueError):
        message = str(exc)
        if not re.search(r"(?:[A-Za-z]:\\|/[^ ]+/)", message):
            for variable in ("GOOGLE_API_KEY", "GEMINI_API_KEY"):
                secret = os.environ.get(variable)
                if secret:
                    message = message.replace(secret, "[REDACTED]")
            return ValueError(f"{operation}: {message}")
    return ValueError(f"{operation}: fallo controlado ({type(exc).__name__})")


def _model_identity(summary: dict[str, Any]) -> ModelIdentity:
    champion = summary.get("champion_final", {})
    registry = champion.get("registry", {})
    version = registry.get("version")
    return ModelIdentity(
        name=str(registry.get("name") or "credit-approval-v2"),
        version=None if version is None else str(version),
        selected_configuration=champion.get("selected_from"),
        dataset_version=summary.get("dataset_version"),
    )


def predict_credit_application_impl(
    application: CreditApplication,
    resources: MCPResourceManager,
) -> PredictionOutput:
    try:
        payload = application.as_model_input()
        pipeline, summary = resources.model_resources()
        frame = pd.DataFrame([payload], columns=MODEL_PREDICTOR_COLUMNS)
        estimator = pipeline.named_steps["estimador"]
        positive_index = localizar_clase_positiva(estimator)
        prediction = int(pipeline.predict(frame)[0])
        probability = float(pipeline.predict_proba(frame)[0, positive_index])
        return PredictionOutput(
            prediction=prediction,
            prediction_label="approved" if prediction == 1 else "denied",
            probability=probability,
            model=_model_identity(summary),
            warnings=[
                "Historical approval prediction; not a default-risk estimate or autonomous decision."
            ],
        )
    except Exception as exc:
        raise _controlled_error("prediction failed", exc) from None


def explain_credit_application_impl(
    application: CreditApplication,
    resources: MCPResourceManager,
) -> ExplanationOutput:
    try:
        payload = application.as_model_input()
        pipeline, summary = resources.model_resources()
        evidence = _json_safe(
            explicar_solicitud(pipeline, payload, top_n=5, summary=summary)
        )
        return ExplanationOutput(
            prediction=evidence["prediction"],
            prediction_label=evidence["prediction_label"],
            probability=evidence["probability"],
            shap_output_space=evidence["shap_output_space"],
            top_positive_factors=evidence["top_positive_factors"],
            top_negative_factors=evidence["top_negative_factors"],
            model=evidence["model"],
            dataset=evidence["dataset"],
            non_causality_warning=evidence["warning"],
            evidence_package=evidence,
        )
    except Exception as exc:
        raise _controlled_error("explanation failed", exc) from None


def retrieve_credit_policy_impl(
    query: str,
    top_k: int,
    resources: MCPResourceManager,
) -> PolicyRetrievalOutput:
    try:
        embedder, qdrant_client = resources.rag_resources()
        retrieval = retrieve_policy_context(
            query,
            k=top_k,
            collection_name=COLLECTION_NAME,
            embedder=embedder,
            client=qdrant_client,
        )
        return PolicyRetrievalOutput(
            query=retrieval["query"],
            top_k=top_k,
            results=[PolicyResult.model_validate(item) for item in retrieval["results"]],
            warnings=[
                "Retrieval scores rank corpus chunks; they are not calibrated confidence or eligibility rules."
            ],
        )
    except Exception as exc:
        raise _controlled_error("policy retrieval failed", exc) from None


def analyze_credit_application_impl(
    application: CreditApplication,
    resources: MCPResourceManager,
) -> AnalysisOutput:
    try:
        state = _json_safe(resources.invoke_workflow(application.as_model_input()))
        response = GroundedResponse.model_validate(state["structured_response"])
        return AnalysisOutput(
            prediction=state.get("prediction"),
            prediction_label=state.get("prediction_label"),
            probability=state.get("probability"),
            xai_evidence=state.get("xai_evidence"),
            rag_queries=state.get("rag_queries", []),
            policy_evidence=state.get("policy_evidence", []),
            evidence_sufficiency=state.get("evidence_sufficiency"),
            grounded_response=response,
            final_response=state["final_response"],
            sources=[GroundedSource.model_validate(item) for item in state.get("sources", [])],
            warnings=list(state.get("warnings", [])),
            errors=list(state.get("errors", [])),
            node_trace=list(state.get("node_trace", [])),
            gemini_called=bool(state.get("gemini_called", False)),
            generation_attempts=int(state.get("generation_attempts", 0)),
            generation_status=str(state.get("generation_status", "unknown")),
            provider_status=str(state.get("provider_status", "unknown")),
            citation_normalization_count=int(
                state.get("citation_normalization_count", 0)
            ),
        )
    except Exception as exc:
        raise _controlled_error("grounded analysis failed", exc) from None


READ_ONLY = ToolAnnotations(
    readOnlyHint=True,
    destructiveHint=False,
    idempotentHint=True,
    openWorldHint=False,
)


def create_mcp_server(
    resource_factory: Callable[[], MCPResourceManager] = MCPResourceManager,
) -> MCPServer[MCPRuntime]:
    @asynccontextmanager
    async def lifespan(_: MCPServer[MCPRuntime]):
        resources = resource_factory()
        try:
            yield MCPRuntime(resources=resources)
        finally:
            resources.close()

    server: MCPServer[MCPRuntime] = MCPServer(
        SERVER_NAME,
        title="Credit Approval V2 MCP",
        description=(
            "Read-only tools over the existing HMDA V2 CatBoost, SHAP, policy "
            "retrieval and deterministic grounded LangGraph workflow."
        ),
        version=SERVER_VERSION,
        lifespan=lifespan,
    )

    @server.tool(annotations=READ_ONLY)
    def predict_credit_application(
        application: CreditApplication,
        ctx: Context[MCPRuntime],
    ) -> PredictionOutput:
        """Predict historical HMDA loan approval from exactly 16 raw V2 fields."""
        return predict_credit_application_impl(
            application, ctx.request_context.lifespan_context.resources
        )

    @server.tool(annotations=READ_ONLY)
    def explain_credit_application(
        application: CreditApplication,
        ctx: Context[MCPRuntime],
    ) -> ExplanationOutput:
        """Return the existing non-causal local Tree SHAP evidence package."""
        return explain_credit_application_impl(
            application, ctx.request_context.lifespan_context.resources
        )

    @server.tool(annotations=READ_ONLY)
    def retrieve_credit_policy(
        query: str = Field(min_length=3, max_length=500),
        top_k: int = Field(default=3, ge=1, le=10),
        ctx: Context[MCPRuntime] = None,  # type: ignore[assignment]
    ) -> PolicyRetrievalOutput:
        """Retrieve bounded official mortgage-policy chunks from local Qdrant."""
        if ctx is None:  # pragma: no cover - MCP siempre inyecta Context
            raise ValueError("MCP context unavailable")
        return retrieve_credit_policy_impl(
            query,
            top_k,
            ctx.request_context.lifespan_context.resources,
        )

    @server.tool(annotations=READ_ONLY)
    def analyze_credit_application(
        application: CreditApplication,
        ctx: Context[MCPRuntime],
    ) -> AnalysisOutput:
        """Run the existing deterministic ML→XAI→RAG→guardrail→Gemini graph."""
        return analyze_credit_application_impl(
            application, ctx.request_context.lifespan_context.resources
        )

    return server


mcp = create_mcp_server()


class _FixtureGeminiModels:
    def __init__(self, parsed: dict[str, Any]) -> None:
        self._parsed = parsed

    def generate_content(self, **_: Any) -> Any:
        response = dict(self._parsed)
        response["cited_source_ids"] = [
            source["citation_id"] for source in response.get("sources", [])
        ]
        for key in ("prediction", "probability", "sources", "warnings"):
            response.pop(key, None)
        return SimpleNamespace(parsed=response)


class _FixtureGeminiClient:
    def __init__(self, parsed: dict[str, Any]) -> None:
        self.models = _FixtureGeminiModels(parsed)


def _example_application() -> dict[str, Any]:
    data = pd.read_csv(DEFAULT_DATASET_PATH, nrows=1)
    payload = data.loc[0, list(MODEL_PREDICTOR_COLUMNS)].to_dict()
    return _json_safe(payload)


async def generate_mcp_artifacts(
    output_dir: Path = DEFAULT_MCP_ARTIFACT_DIR,
) -> dict[str, Any]:
    """Genera manifest y ejemplos mediante un cliente MCP real in-process."""
    grounded_fixture = json.loads(DEFAULT_GROUNDED_FIXTURE.read_text(encoding="utf-8"))
    resources = MCPResourceManager(
        gemini_client=_FixtureGeminiClient(grounded_fixture),
        load_environment=False,
    )
    server = create_mcp_server(lambda: resources)
    application = _example_application()
    async with Client(server) as client:
        listed = await client.list_tools()
        manifest = {
            "tools": [tool.model_dump(mode="json", by_alias=True) for tool in listed.tools]
        }
        calls = {
            "example_predict.json": await client.call_tool(
                "predict_credit_application", {"application": application}
            ),
            "example_xai.json": await client.call_tool(
                "explain_credit_application", {"application": application}
            ),
            "example_retrieval.json": await client.call_tool(
                "retrieve_credit_policy",
                {"query": "mortgage debt-to-income ratio", "top_k": 3},
            ),
            "example_grounded_analysis.json": await client.call_tool(
                "analyze_credit_application", {"application": application}
            ),
        }
    if tuple(tool["name"] for tool in manifest["tools"]) != TOOL_NAMES:
        raise AssertionError("MCP tool manifest no coincide con el contrato de cuatro tools")
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "server_name": SERVER_NAME,
        "server_version": SERVER_VERSION,
        "sdk": "mcp",
        "sdk_version": package_version("mcp"),
        "transport": TRANSPORT,
        "tool_count": len(manifest["tools"]),
        "resource_lifecycle": "lazy_per_stdio_session",
        "generated_at": datetime.now(UTC).isoformat(),
        "grounded_example_provider": "validated_fixture_replayed_through_existing_workflow",
        "audit_only_excluded": list(AUDIT_ONLY_COLUMNS),
        "target_excluded": TARGET_COLUMN,
        "secrets_recorded": False,
    }
    (output_dir / "server_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "tools_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    for filename, result in calls.items():
        if result.is_error or result.structured_content is None:
            raise RuntimeError(f"La llamada MCP para {filename} fallo")
        (output_dir / filename).write_text(
            json.dumps(
                result.structured_content,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--generate-artifacts",
        action="store_true",
        help="Genera manifest y ejemplos locales sin realizar una llamada live a Gemini.",
    )
    args = parser.parse_args()
    if args.generate_artifacts:
        asyncio.run(generate_mcp_artifacts())
        return
    mcp.run(transport=TRANSPORT)


if __name__ == "__main__":
    main()
