"""Tests unitarios del workflow determinista ML/XAI/RAG/Gemini."""

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from google.genai import errors as genai_errors

from src.construir_dataset_v2 import AUDIT_ONLY_COLUMNS, MODEL_PREDICTOR_COLUMNS
from src.langgraph_v2 import (
    DEFAULT_GENAI_MODEL,
    NODE_ORDER,
    GenAIConfig,
    GroundedCreditWorkflow,
    WorkflowResources,
    _normalize_citation_id,
    _write_json,
    cargar_config_genai,
    construir_prompt_grounded,
)


def _application() -> dict[str, object]:
    return {column: 1 for column in MODEL_PREDICTOR_COLUMNS}


def _xai_evidence() -> dict[str, object]:
    return {
        "prediction": 1,
        "prediction_label": "approved",
        "probability": 0.8,
        "shap_output_space": "CatBoost raw margin (log-odds)",
        "top_positive_factors": [
            {
                "model_feature": "dti_category",
                "source_features": ["debt_to_income_ratio"],
                "shap_value": 0.4,
            }
        ],
        "top_negative_factors": [],
        "warning": "SHAP describe asociaciones del modelo, no causalidad.",
    }


def _retrieval() -> dict[str, object]:
    return {
        "query": "HMDA debt-to-income ratio underwriting guidance",
        "results": [
            {
                "text": "The debt-to-income ratio is a reported underwriting field.",
                "score": 0.72,
                "source_id": "fannie-selling-guide",
                "source_title": "Fannie Mae Selling Guide",
                "institution": "Fannie Mae",
                "source_url": "https://selling-guide.fanniemae.com/example",
                "page_or_section": "B3-6-02",
                "chunk_id": "fannie-selling-guide-001",
            }
        ],
    }


class _FakeEstimator:
    classes_ = np.array([0, 1])


class _FakePipeline:
    def __init__(self) -> None:
        self.named_steps = {"estimador": _FakeEstimator()}
        self.predict_calls = 0

    def predict(self, frame):
        self.predict_calls += 1
        return np.array([1])

    def predict_proba(self, frame):
        return np.array([[0.2, 0.8]])


class _FakeModels:
    def __init__(self, parsed: dict[str, object] | None = None) -> None:
        self.calls: list[dict[str, object]] = []
        self.parsed = parsed or {
            "model_explanation": "El modelo asocio el perfil con aprobacion.",
            "policy_context": "La guia describe el DTI como contexto [S1].",
            "final_assessment": "Prediccion y politica son evidencias distintas [S1].",
            "cited_source_ids": ["[S1]"],
        }

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(parsed=self.parsed, text=None)


class _FakeGeminiClient:
    def __init__(self, parsed: dict[str, object] | None = None) -> None:
        self.models = _FakeModels(parsed)


def _workflow(
    client: _FakeGeminiClient | None = None,
    genai_config: GenAIConfig | None = None,
):
    pipeline = _FakePipeline()
    client = client or _FakeGeminiClient()
    resources = WorkflowResources(
        pipeline=pipeline,
        summary={},
        embedder=object(),
        qdrant_client=object(),
        genai_config=genai_config
        or GenAIConfig(provider="gemini", model=DEFAULT_GENAI_MODEL),
        gemini_client=client,
        rag_top_k=3,
    )
    return GroundedCreditWorkflow(resources), pipeline, client


def _generation_state() -> dict[str, object]:
    return {
        "prediction_label": "approved",
        "probability": 0.8,
        "xai_evidence": _xai_evidence(),
        "policy_evidence": [{"results": [{"citation_id": "[S1]"}]}],
        "sources": [
            {
                "citation_id": "[S1]",
                "source_id": "official",
                "source_title": "Official",
                "institution": "Fannie Mae",
                "source_url": "https://selling-guide.fanniemae.com/example",
                "page_or_section": "B3-6-02",
                "chunk_id": "chunk-1",
            }
        ],
        "insufficient_evidence": False,
        "warnings": [],
        "errors": [],
    }


class _SequenceModels:
    def __init__(self, outcomes: list[object]) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[dict[str, object]] = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        if isinstance(outcome, str):
            return SimpleNamespace(parsed=None, text=outcome)
        return SimpleNamespace(parsed=outcome, text=None)


class _SequenceClient:
    def __init__(self, outcomes: list[object]) -> None:
        self.models = _SequenceModels(outcomes)


def _success_payload() -> dict[str, object]:
    return {
        "model_explanation": "El modelo asocio el perfil con aprobacion.",
        "policy_context": "La guia describe el DTI como contexto [S1].",
        "final_assessment": "Prediccion y politica son evidencias distintas [S1].",
        "cited_source_ids": ["[S1]"],
    }


def _citation_state(*numbers: int) -> dict[str, object]:
    state = _generation_state()
    state["sources"] = [
        {
            "citation_id": f"[S{number}]",
            "source_id": f"official-{number}",
            "source_title": f"Official {number}",
            "institution": "Fannie Mae",
            "source_url": "https://selling-guide.fanniemae.com/example",
            "page_or_section": f"section-{number}",
            "chunk_id": f"chunk-{number}",
        }
        for number in numbers
    ]
    return state


def _citation_payload(
    citation_ids: list[str],
    *,
    policy_context: str = "Contexto documental suministrado.",
) -> dict[str, object]:
    return {
        "model_explanation": "Asociacion interna del modelo, no causal.",
        "policy_context": policy_context,
        "final_assessment": "Prediccion y contexto se mantienen separados.",
        "cited_source_ids": citation_ids,
    }


class TestLangGraphV2(unittest.TestCase):
    def test_unbracketed_declared_citations_are_normalized_and_allowed(self):
        client = _SequenceClient([_citation_payload(["S3", "S5"])])
        workflow, _, _ = _workflow(client)

        result = workflow._generate_grounded_response(_citation_state(3, 5))

        self.assertEqual(result["generation_status"], "success")
        self.assertEqual(result["citation_normalization_count"], 2)
        self.assertEqual(
            [source["citation_id"] for source in result["structured_response"]["sources"]],
            ["[S3]", "[S5]"],
        )

    def test_canonical_declared_citations_remain_valid_without_normalization(self):
        client = _SequenceClient([_citation_payload(["[S3]", "[S5]"])])
        workflow, _, _ = _workflow(client)

        result = workflow._generate_grounded_response(_citation_state(3, 5))

        self.assertEqual(result["generation_status"], "success")
        self.assertEqual(result["citation_normalization_count"], 0)

    def test_lowercase_and_spaced_citations_are_normalized(self):
        client = _SequenceClient([_citation_payload([" s3 ", "[ s5 ]"])])
        workflow, _, _ = _workflow(client)

        result = workflow._generate_grounded_response(_citation_state(3, 5))

        self.assertEqual(result["generation_status"], "success")
        self.assertEqual(result["citation_normalization_count"], 2)
        self.assertEqual(
            [source["citation_id"] for source in result["structured_response"]["sources"]],
            ["[S3]", "[S5]"],
        )

    def test_unknown_unbracketed_citation_still_fails_allowlist(self):
        client = _SequenceClient([_citation_payload(["S99"])])
        workflow, _, _ = _workflow(client)

        result = workflow._generate_grounded_response(_citation_state(3))

        self.assertEqual(result["generation_status"], "output_validation_failed")
        self.assertEqual(result["citation_normalization_count"], 1)
        self.assertEqual(result["structured_response"]["sources"], [])
        self.assertIn("[S99]", result["structured_response"]["final_assessment"])

    def test_unbracketed_citation_in_free_text_is_detected_and_normalized(self):
        client = _SequenceClient(
            [_citation_payload([], policy_context="Contexto respaldado por S3.")]
        )
        workflow, _, _ = _workflow(client)

        result = workflow._generate_grounded_response(_citation_state(3))

        self.assertEqual(result["generation_status"], "success")
        self.assertEqual(result["citation_normalization_count"], 1)
        self.assertIn("[S3]", result["structured_response"]["policy_context"])

    def test_non_citation_strings_are_not_auto_corrected(self):
        for raw in ("source 3", "ref 3", "3", "XS3", "S3extra"):
            with self.subTest(raw=raw):
                self.assertEqual(_normalize_citation_id(raw), raw)

    def test_default_genai_model_is_flash_lite(self):
        with patch.dict(os.environ, {}, clear=True):
            config = cargar_config_genai()

        self.assertEqual(config.provider, "gemini")
        self.assertEqual(config.model, "gemini-3.5-flash-lite")

    def test_genai_model_can_be_overridden_from_environment(self):
        with patch.dict(
            os.environ,
            {"GENAI_PROVIDER": "gemini", "GENAI_MODEL": "configured-model-id"},
            clear=True,
        ):
            config = cargar_config_genai()
        workflow, _, client = _workflow(genai_config=config)

        workflow._generate_grounded_response(_generation_state())

        self.assertEqual(config.model, "configured-model-id")
        self.assertEqual(client.models.calls[0]["model"], "configured-model-id")

    def test_api_key_is_redacted_from_state_and_serialized_artifact(self):
        secret = "unit-test-secret-must-not-leak"
        client = _SequenceClient([RuntimeError(f"provider rejected {secret}")])
        workflow, _, _ = _workflow(client)
        with patch.dict(os.environ, {"GOOGLE_API_KEY": secret}):
            result = workflow._generate_grounded_response(_generation_state())
            serialized_state = json.dumps(result, allow_nan=False)
            with TemporaryDirectory() as directory:
                artifact = Path(directory) / "state.json"
                _write_json(artifact, result)
                serialized_artifact = artifact.read_text(encoding="utf-8")

        self.assertNotIn(secret, serialized_state)
        self.assertNotIn(secret, serialized_artifact)
        self.assertIn("[REDACTED]", serialized_artifact)

    def test_node_order_is_deterministic_and_existing_components_are_reused(self):
        workflow, _, client = _workflow()
        xai = _xai_evidence()
        queries = {
            "queries": [
                {
                    "origin_feature": "dti_category",
                    "query": "HMDA debt-to-income ratio underwriting guidance",
                }
            ],
            "generator": "deterministic_feature_template_v1",
        }
        with (
            patch("src.langgraph_v2.explicar_solicitud", return_value=xai) as explain,
            patch(
                "src.langgraph_v2.construir_queries_desde_evidence",
                return_value=queries,
            ) as build_queries,
            patch(
                "src.langgraph_v2.retrieve_policy_context",
                return_value=_retrieval(),
            ) as retrieve,
        ):
            result = workflow.invoke(_application())

        self.assertEqual(result["node_trace"], list(NODE_ORDER))
        explain.assert_called_once()
        build_queries.assert_called_once_with(xai)
        retrieve.assert_called_once()
        self.assertIs(retrieve.call_args.kwargs["embedder"], workflow.resources.embedder)
        self.assertIs(retrieve.call_args.kwargs["client"], workflow.resources.qdrant_client)
        self.assertEqual(len(client.models.calls), 1)

    def test_audit_only_input_fails_closed_before_prediction(self):
        workflow, pipeline, client = _workflow()
        application = _application()
        application[AUDIT_ONLY_COLUMNS[0]] = "audit"
        result = workflow.invoke(application)

        self.assertTrue(result["insufficient_evidence"])
        self.assertEqual(pipeline.predict_calls, 0)
        self.assertFalse(result["gemini_called"])
        self.assertIn("audit-only", " ".join(result["errors"]))
        self.assertEqual(result["structured_response"]["sources"], [])
        self.assertNotIn("xai_evidence", result)
        self.assertEqual(len(client.models.calls), 0)

    def test_guardrail_accepts_complete_official_evidence_covering_xai_factor(self):
        workflow, _, _ = _workflow()
        state = {
            "xai_evidence": _xai_evidence(),
            "rag_queries": [{"origin_feature": "dti_category", "query": "dti"}],
            "policy_evidence": [
                {
                    "origin_feature": "dti_category",
                    "query": "dti",
                    "results": _retrieval()["results"],
                }
            ],
            "warnings": [],
            "errors": [],
        }
        result = workflow._evidence_guardrail(state)

        self.assertFalse(result["insufficient_evidence"])
        self.assertEqual(result["evidence_sufficiency"]["covered_xai_features"], ["dti_category"])
        self.assertEqual(result["sources"][0]["citation_id"], "[S1]")
        self.assertIsNone(result["evidence_sufficiency"]["numeric_score_threshold"])

    def test_insufficient_evidence_abstains_without_calling_gemini(self):
        workflow, _, client = _workflow()
        xai = _xai_evidence()
        queries = {
            "queries": [{"origin_feature": "dti_category", "query": "dti"}],
            "generator": "deterministic_feature_template_v1",
        }
        with (
            patch("src.langgraph_v2.explicar_solicitud", return_value=xai),
            patch("src.langgraph_v2.construir_queries_desde_evidence", return_value=queries),
            patch(
                "src.langgraph_v2.retrieve_policy_context",
                return_value={"query": "dti", "results": []},
            ),
        ):
            result = workflow.invoke(_application())

        self.assertTrue(result["insufficient_evidence"])
        self.assertFalse(result["gemini_called"])
        self.assertIn("Abstencion controlada", result["structured_response"]["final_assessment"])
        self.assertEqual(result["structured_response"]["sources"], [])
        self.assertEqual(len(client.models.calls), 0)

    def test_prompt_separates_evidence_and_forbids_causal_claims(self):
        prompt = construir_prompt_grounded(_xai_evidence(), [], ["warning"])
        self.assertIn("MODEL_EVIDENCE:", prompt)
        self.assertIn("POLICY_EVIDENCE:", prompt)
        self.assertIn("WARNINGS:", prompt)
        self.assertIn("Do not claim causality", prompt)
        self.assertIn("not direct changes in probability", prompt)
        self.assertIn("Do not invent rules, thresholds, facts, or sources", prompt)
        self.assertIn("distinguish the ML prediction", prompt)

    def test_generated_citations_are_limited_to_retrieved_evidence(self):
        workflow, _, client = _workflow()
        state = _generation_state()
        result = workflow._generate_grounded_response(state)

        self.assertTrue(result["gemini_called"])
        self.assertEqual(result["structured_response"]["sources"], state["sources"])
        json.dumps(result["structured_response"], allow_nan=False)
        self.assertEqual(len(client.models.calls), 1)

    def test_503_retries_then_succeeds(self):
        unavailable = genai_errors.ServerError(
            503,
            {"error": {"code": 503, "status": "UNAVAILABLE", "message": "busy"}},
        )
        client = _SequenceClient([unavailable, _success_payload()])
        workflow, _, _ = _workflow(client)
        sleep = Mock()
        workflow.resources.sleep_fn = sleep

        result = workflow._generate_grounded_response(_generation_state())

        self.assertEqual(result["generation_status"], "success")
        self.assertEqual(result["provider_status"], "success")
        self.assertEqual(result["generation_attempts"], 2)
        self.assertEqual(len(result["generation_errors"]), 1)
        self.assertEqual(len(client.models.calls), 2)
        sleep.assert_called_once_with(1.0)

    def test_truncated_json_retries_then_succeeds(self):
        client = _SequenceClient(['{"model_explanation": "truncated', _success_payload()])
        workflow, _, _ = _workflow(client)
        sleep = Mock()
        workflow.resources.sleep_fn = sleep

        result = workflow._generate_grounded_response(_generation_state())

        self.assertEqual(result["generation_status"], "success")
        self.assertEqual(result["generation_attempts"], 2)
        self.assertEqual(result["generation_errors"][0]["error_type"], "JSONDecodeError")
        self.assertTrue(result["generation_errors"][0]["retryable"])
        sleep.assert_called_once_with(1.0)

    def test_three_retryable_failures_produce_fail_closed_abstention(self):
        failures = [
            genai_errors.ServerError(
                503,
                {"error": {"code": 503, "status": "UNAVAILABLE", "message": "busy"}},
            )
            for _ in range(3)
        ]
        client = _SequenceClient(failures)
        workflow, _, _ = _workflow(client)
        sleep = Mock()
        workflow.resources.sleep_fn = sleep

        result = workflow._generate_grounded_response(_generation_state())

        self.assertTrue(result["gemini_called"])
        self.assertEqual(result["generation_status"], "provider_unavailable")
        self.assertEqual(result["provider_status"], "unavailable")
        self.assertEqual(result["generation_attempts"], 3)
        self.assertEqual(len(result["generation_errors"]), 3)
        self.assertEqual(result["structured_response"]["sources"], [])
        self.assertIn("Abstencion controlada", result["structured_response"]["final_assessment"])
        self.assertIn("proveedor GenAI", result["structured_response"]["policy_context"])
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [1.0, 2.0])

    def test_auth_error_is_not_retried(self):
        auth_error = genai_errors.ClientError(
            401,
            {"error": {"code": 401, "status": "UNAUTHENTICATED", "message": "bad key"}},
        )
        client = _SequenceClient([auth_error, _success_payload()])
        workflow, _, _ = _workflow(client)
        sleep = Mock()
        workflow.resources.sleep_fn = sleep

        result = workflow._generate_grounded_response(_generation_state())

        self.assertEqual(result["generation_status"], "provider_error")
        self.assertEqual(result["provider_status"], "permanent_error")
        self.assertEqual(result["generation_attempts"], 1)
        self.assertEqual(len(client.models.calls), 1)
        sleep.assert_not_called()

    def test_maximum_three_attempts_is_enforced(self):
        failures = [
            genai_errors.ServerError(
                503,
                {"error": {"code": 503, "status": "UNAVAILABLE", "message": "busy"}},
            )
            for _ in range(4)
        ]
        client = _SequenceClient(failures)
        workflow, _, _ = _workflow(client)
        workflow.resources.sleep_fn = Mock()

        result = workflow._generate_grounded_response(_generation_state())

        self.assertEqual(result["generation_attempts"], 3)
        self.assertEqual(len(client.models.calls), 3)

    def test_unretrieved_citation_triggers_controlled_abstention(self):
        client = _FakeGeminiClient(
            {
                "model_explanation": "Resumen.",
                "policy_context": "Fuente inexistente [S99].",
                "final_assessment": "No debe publicarse.",
                "cited_source_ids": ["[S99]"],
            }
        )
        workflow, _, _ = _workflow(client)
        state = {
            "prediction_label": "approved",
            "probability": 0.8,
            "xai_evidence": _xai_evidence(),
            "policy_evidence": [],
            "sources": [{"citation_id": "[S1]"}],
            "insufficient_evidence": False,
            "warnings": [],
            "errors": [],
        }
        result = workflow._generate_grounded_response(state)

        self.assertIn("Abstencion controlada", result["structured_response"]["final_assessment"])
        self.assertEqual(result["structured_response"]["sources"], [])
        self.assertNotIn("[S99]", result["structured_response"]["policy_context"])

    def test_repeated_workflow_is_reproducible_with_mocked_gemini(self):
        workflow, _, _ = _workflow()
        queries = {
            "queries": [{"origin_feature": "dti_category", "query": "dti"}],
            "generator": "deterministic_feature_template_v1",
        }
        with (
            patch("src.langgraph_v2.explicar_solicitud", return_value=_xai_evidence()),
            patch("src.langgraph_v2.construir_queries_desde_evidence", return_value=queries),
            patch("src.langgraph_v2.retrieve_policy_context", return_value=_retrieval()),
        ):
            first = workflow.invoke(_application())
            second = workflow.invoke(_application())

        self.assertEqual(first["structured_response"], second["structured_response"])
        self.assertEqual(first["node_trace"], second["node_trace"])


if __name__ == "__main__":
    unittest.main()
