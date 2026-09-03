import asyncio
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from mcp.client import Client
from mcp.client.stdio import StdioServerParameters

from src.construir_dataset_v2 import AUDIT_ONLY_COLUMNS, MODEL_PREDICTOR_COLUMNS
from src.mcp_server_v2 import (
    TOOL_NAMES,
    CreditApplication,
    MCPResourceManager,
    analyze_credit_application_impl,
    create_mcp_server,
    explain_credit_application_impl,
    predict_credit_application_impl,
    retrieve_credit_policy_impl,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def application_payload() -> dict[str, object]:
    return {
        "income": 153.0,
        "loan_amount": 245000.0,
        "loan_term": 360.0,
        "loan_purpose": 1,
        "loan_type": 1,
        "lien_status": 1,
        "preapproval": 2,
        "debt_to_income_ratio": "46",
        "combined_loan_to_value_ratio": 90.0,
        "property_value": 265000.0,
        "occupancy_type": 1,
        "construction_method": 1,
        "total_units": 1,
        "submission_of_application": 1,
        "interest_only_payment": 2,
        "balloon_payment": 2,
    }


class FakeEstimator:
    classes_ = np.array([0, 1])


class FakePipeline:
    named_steps = {"estimador": FakeEstimator()}

    def predict(self, _):
        return np.array([1])

    def predict_proba(self, _):
        return np.array([[0.2, 0.8]])


class FakeResources:
    def __init__(self) -> None:
        self.pipeline = FakePipeline()
        self.summary = {
            "dataset_version": "fixture-v2",
            "champion_final": {
                "selected_from": "fixture-catboost",
                "registry": {"name": "credit-approval-v2", "version": "7"},
            },
        }
        self.closed = False
        self.model_calls = 0
        self.rag_calls = 0
        self.workflow_calls = 0

    def model_resources(self):
        self.model_calls += 1
        return self.pipeline, self.summary

    def rag_resources(self):
        self.rag_calls += 1
        return object(), object()

    def invoke_workflow(self, _application):
        self.workflow_calls += 1
        source = {
            "citation_id": "[S1]",
            "source_id": "official",
            "source_title": "Official source",
            "institution": "CFPB",
            "source_url": "https://www.consumerfinance.gov/source",
            "page_or_section": "page 1",
            "chunk_id": "official:0001",
        }
        grounded = {
            "prediction": "approved",
            "probability": 0.8,
            "model_explanation": "Association, not causality.",
            "policy_context": "Official context [S1].",
            "final_assessment": "Analytical support only [S1].",
            "sources": [source],
            "warnings": [],
        }
        return {
            "prediction": 1,
            "prediction_label": "approved",
            "probability": 0.8,
            "xai_evidence": {"warning": "non-causal"},
            "rag_queries": [{"origin_feature": "income", "query": "income"}],
            "policy_evidence": [{"results": [{"chunk_id": "official:0001"}]}],
            "evidence_sufficiency": {"insufficient_evidence": False},
            "structured_response": grounded,
            "final_response": "Grounded result",
            "sources": [source],
            "warnings": [],
            "errors": [],
            "node_trace": ["validate_input", "predict_ml"],
            "gemini_called": True,
            "generation_attempts": 1,
            "generation_status": "success",
            "provider_status": "success",
            "citation_normalization_count": 0,
        }

    def close(self):
        self.closed = True


class MCPContractTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.resources = FakeResources()

    def client_context(self):
        return Client(create_mcp_server(lambda: self.resources), raise_exceptions=False)

    async def test_server_lists_exactly_four_typed_tools(self):
        async with self.client_context() as client:
            listed = await client.list_tools()
        self.assertEqual(tuple(tool.name for tool in listed.tools), TOOL_NAMES)
        for tool in listed.tools:
            self.assertIsNotNone(tool.output_schema)
        predict_schema = listed.tools[0].input_schema
        application_schema = predict_schema["$defs"]["CreditApplication"]
        self.assertFalse(application_schema["additionalProperties"])
        self.assertEqual(
            set(application_schema["properties"]), set(MODEL_PREDICTOR_COLUMNS)
        )
        serialized = json.dumps([tool.model_dump() for tool in listed.tools])
        for column in AUDIT_ONLY_COLUMNS:
            self.assertNotIn(column, serialized)
        self.assertNotIn("LoanApproved", serialized)

    async def test_invalid_and_audit_only_inputs_are_rejected(self):
        missing = application_payload()
        missing.pop("income")
        async with self.client_context() as client:
            result = await client.call_tool(
                "predict_credit_application", {"application": missing}
            )
        self.assertTrue(result.is_error)

        forbidden = application_payload() | {AUDIT_ONLY_COLUMNS[0]: "55-64"}
        async with self.client_context() as client:
            result = await client.call_tool(
                "predict_credit_application", {"application": forbidden}
            )
        self.assertTrue(result.is_error)

    async def test_prediction_is_structured_and_secret_free(self):
        secret = "never-return-this-api-key"
        with patch.dict(os.environ, {"GOOGLE_API_KEY": secret}):
            async with self.client_context() as client:
                result = await client.call_tool(
                    "predict_credit_application", {"application": application_payload()}
                )
        self.assertFalse(result.is_error)
        self.assertEqual(result.structured_content["prediction"], 1)
        self.assertEqual(result.structured_content["probability"], 0.8)
        self.assertNotIn(secret, json.dumps(result.structured_content))
        json.dumps(result.structured_content, allow_nan=False)

    async def test_langgraph_tool_uses_existing_workflow_result(self):
        async with self.client_context() as client:
            result = await client.call_tool(
                "analyze_credit_application", {"application": application_payload()}
            )
        self.assertFalse(result.is_error)
        self.assertEqual(result.structured_content["generation_status"], "success")
        self.assertEqual(result.structured_content["sources"][0]["citation_id"], "[S1]")
        self.assertEqual(self.resources.workflow_calls, 1)


class MCPImplementationTests(unittest.TestCase):
    def setUp(self):
        self.application = CreditApplication.model_validate(application_payload())
        self.resources = FakeResources()

    def test_predict_wrapper_reuses_champion_contract(self):
        result = predict_credit_application_impl(self.application, self.resources)
        self.assertEqual(result.prediction_label, "approved")
        self.assertEqual(result.model.selected_configuration, "fixture-catboost")

    def test_xai_wrapper_reuses_explicar_solicitud(self):
        evidence = {
            "prediction": 1,
            "prediction_label": "approved",
            "probability": 0.8,
            "base_value": 0.1,
            "shap_output_space": "raw margin",
            "top_positive_factors": [{"model_feature": "income", "shap_value": 0.2}],
            "top_negative_factors": [],
            "global_feature_context": None,
            "model": {"name": "credit-approval-v2", "version": "7"},
            "dataset": {"version": "fixture-v2", "sha256": "abc"},
            "warning": "SHAP is associative, not causal.",
        }
        with patch("src.mcp_server_v2.explicar_solicitud", return_value=evidence) as call:
            result = explain_credit_application_impl(self.application, self.resources)
        call.assert_called_once()
        self.assertEqual(result.top_positive_factors[0]["model_feature"], "income")
        self.assertIn("not causal", result.non_causality_warning)

    def test_rag_wrapper_reuses_existing_retriever(self):
        payload = {
            "query": "mortgage income",
            "results": [
                {
                    "text": "official text",
                    "score": 0.9,
                    "source_id": "official",
                    "source_title": "Official source",
                    "institution": "CFPB",
                    "source_url": "https://www.consumerfinance.gov/source",
                    "page_or_section": "page 1",
                    "chunk_id": "official:0001",
                }
            ],
        }
        with patch(
            "src.mcp_server_v2.retrieve_policy_context", return_value=payload
        ) as call:
            result = retrieve_credit_policy_impl(
                "mortgage income", 3, self.resources
            )
        call.assert_called_once()
        self.assertEqual(result.results[0].source_id, "official")

    def test_analysis_output_is_json_serializable(self):
        result = analyze_credit_application_impl(self.application, self.resources)
        json.dumps(result.model_dump(mode="json"), allow_nan=False)

    def test_resource_manager_loads_each_resource_once(self):
        pipeline = FakePipeline()
        summary = self.resources.summary
        manager = MCPResourceManager(load_environment=False)
        with (
            patch("src.mcp_server_v2.cargar_champion", return_value=pipeline) as champion,
            patch("src.mcp_server_v2.cargar_resumen", return_value=summary) as resumen,
            patch("src.mcp_server_v2.LocalSentenceEmbedder", return_value=object()) as embedder,
            patch("src.mcp_server_v2.QdrantClient", return_value=object()) as qdrant,
        ):
            manager.model_resources()
            manager.model_resources()
            manager.rag_resources()
            manager.rag_resources()
        champion.assert_called_once()
        resumen.assert_called_once()
        embedder.assert_called_once()
        qdrant.assert_called_once()


class MCPStdioIntegrationTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls._cache_directory = tempfile.TemporaryDirectory(
            prefix="proyecto-mle-mcp-test-"
        )
        cls.cache_root = Path(cls._cache_directory.name).resolve()
        cls.hf_home = cls.cache_root / "huggingface"
        cls.mpl_config = cls.cache_root / "matplotlib"
        cls.hf_home.mkdir()
        cls.mpl_config.mkdir()

    @classmethod
    def tearDownClass(cls):
        cls._cache_directory.cleanup()

    async def _run_real_stdio_client(self, *, offline: bool):
        env = {
            "PYTHONPATH": str(PROJECT_ROOT),
            "PYTHONUTF8": "1",
            "HF_HOME": str(self.hf_home),
            "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
            "HF_HUB_OFFLINE": "1" if offline else "0",
            "TRANSFORMERS_OFFLINE": "1" if offline else "0",
            "MPLCONFIGDIR": str(self.mpl_config),
        }
        if os.name == "nt":
            env["WINDIR"] = os.environ["WINDIR"]
            env["SYSTEMROOT"] = os.environ["SYSTEMROOT"]
        parameters = StdioServerParameters(
            command=sys.executable,
            args=["-m", "src.mcp_server_v2"],
            cwd=PROJECT_ROOT,
            env=env,
        )
        outputs = []
        async with Client(parameters, read_timeout_seconds=300) as client:
            listed = await client.list_tools()
            self.assertEqual(tuple(tool.name for tool in listed.tools), TOOL_NAMES)
            calls = (
                ("predict_credit_application", {"application": application_payload()}),
                ("explain_credit_application", {"application": application_payload()}),
                (
                    "retrieve_credit_policy",
                    {"query": "mortgage debt-to-income ratio", "top_k": 2},
                ),
            )
            for name, arguments in calls:
                result = await client.call_tool(name, arguments)
                self.assertFalse(result.is_error, msg=str(result.content))
                self.assertIsNotNone(result.structured_content)
                json.dumps(result.structured_content, allow_nan=False)
                outputs.append(result.structured_content)
        return outputs

    async def test_real_stdio_client_discovers_and_executes_three_tools(self):
        self.assertEqual(list(self.hf_home.iterdir()), [])
        self.assertFalse(self.cache_root.is_relative_to(PROJECT_ROOT))
        self.assertFalse((PROJECT_ROOT / ".hf-test-cache").exists())

        first_outputs = await self._run_real_stdio_client(offline=False)
        cached_files = [path for path in self.hf_home.rglob("*") if path.is_file()]
        self.assertTrue(cached_files, "La primera ejecucion no poblo HF_HOME")

        second_outputs = await self._run_real_stdio_client(offline=True)
        self.assertEqual(first_outputs, second_outputs)


if __name__ == "__main__":
    unittest.main()
