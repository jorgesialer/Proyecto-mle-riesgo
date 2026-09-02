import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient

from src.construir_dataset_v2 import AUDIT_ONLY_COLUMNS
from src.rag_v2 import (
    CORPUS_SOURCES,
    EMBEDDING_DIMENSION,
    OFFICIAL_HOSTS,
    TextChunk,
    TextSection,
    LocalSentenceEmbedder,
    adquirir_corpus,
    build_index,
    construir_queries_desde_evidence,
    crear_chunks,
    evaluar_retrieval,
    extraer_corpus,
    extraer_html,
    retrieve_policy_context,
    validar_fuente_oficial,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class KeywordEmbedder:
    model_name = "test-keyword-embedder"
    dimension = 3

    def encode(self, texts):
        rows = []
        for text in texts:
            lowered = text.lower()
            vector = np.array(
                [
                    lowered.count("income"),
                    lowered.count("debt") + lowered.count("dti"),
                    lowered.count("ltv") + lowered.count("loan-to-value"),
                ],
                dtype=np.float32,
            )
            if not vector.any():
                vector[0] = 0.01
            rows.append(vector / np.linalg.norm(vector))
        return np.vstack(rows)


def sample_chunks():
    return [
        TextChunk(
            chunk_id="income:0000",
            source_id="income",
            source_title="Income guide",
            institution="Official institution",
            source_url="https://official.example/income",
            page_or_section="Income stability",
            text="Stable and predictable income must be documented.",
        ),
        TextChunk(
            chunk_id="dti:0000",
            source_id="dti",
            source_title="DTI guide",
            institution="Official institution",
            source_url="https://official.example/dti",
            page_or_section="Debt-to-income",
            text="Debt-to-income DTI includes monthly debt obligations.",
        ),
        TextChunk(
            chunk_id="ltv:0000",
            source_id="ltv",
            source_title="LTV guide",
            institution="Official institution",
            source_url="https://official.example/ltv",
            page_or_section="Loan-to-value",
            text="LTV is the mortgage loan-to-value ratio.",
        ),
    ]


class TestCorpus(unittest.TestCase):
    def test_sources_are_explicitly_official_and_https(self):
        self.assertEqual(len(CORPUS_SOURCES), 5)
        for source in CORPUS_SOURCES:
            validar_fuente_oficial(source)
            self.assertTrue(source.source_url.startswith("https://"))
            self.assertTrue(any(host in source.source_url for host in OFFICIAL_HOSTS))
            self.assertTrue(source.document_date)
            self.assertTrue(source.institution)

    def test_acquired_metadata_matches_local_hashes(self):
        metadata_path = PROJECT_ROOT / "data" / "rag_sources" / "corpus_metadata.json"
        documents = adquirir_corpus(metadata_path=metadata_path)
        self.assertEqual(len(documents), len(CORPUS_SOURCES))
        for item in documents:
            local_path = PROJECT_ROOT / "data" / "rag_sources" / item["local_filename"]
            digest = hashlib.sha256(local_path.read_bytes()).hexdigest()
            self.assertEqual(item["sha256"], digest)
            self.assertTrue(item["retrieved_at"])
            self.assertGreater(item["bytes"], 0)

    def test_html_extraction_retains_heading_and_content(self):
        source = CORPUS_SOURCES[2]
        html = """
        <html><body><nav>Navigation noise</nav><main>
        <h1>Income rules</h1><p>Stable income is required.</p>
        <h2>History</h2><p>Review the earnings history.</p>
        </main></body></html>
        """
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.html"
            path.write_text(html, encoding="utf-8")
            sections = extraer_html(path, source)
        self.assertEqual([item.page_or_section for item in sections], ["Income rules", "History"])
        self.assertNotIn("Navigation noise", " ".join(item.text for item in sections))

    def test_html_nested_paragraph_in_list_item_is_not_duplicated(self):
        source = CORPUS_SOURCES[2]
        html = """
        <html><body><main><h1>Rules</h1>
        <ul><li><p>texto</p></li></ul>
        </main></body></html>
        """
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested-list.html"
            path.write_text(html, encoding="utf-8")
            sections = extraer_html(path, source)
        extracted = " ".join(item.text for item in sections)
        self.assertEqual(extracted.count("texto"), 1)

    def test_real_corpus_extraction_has_traceable_sections(self):
        sections = extraer_corpus()
        self.assertGreater(len(sections), len(CORPUS_SOURCES))
        self.assertEqual({item.source_id for item in sections}, {item.source_id for item in CORPUS_SOURCES})
        self.assertTrue(all(item.page_or_section and item.source_url for item in sections))

    def test_chunking_is_reproducible_and_bounded(self):
        section = TextSection(
            source_id="source",
            title="Title",
            institution="Institution",
            source_url="https://official.example",
            page_or_section="section 1",
            text=" ".join(f"word{index}" for index in range(250)),
        )
        first = crear_chunks([section], chunk_size_chars=180, chunk_overlap_chars=30)
        second = crear_chunks([section], chunk_size_chars=180, chunk_overlap_chars=30)
        self.assertEqual(first, second)
        self.assertGreater(len(first), 1)
        self.assertTrue(all(len(item.text) <= 180 for item in first))
        self.assertEqual(first[0].chunk_id, "source:0000")


class TestEmbeddingsAndQdrant(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = QdrantClient(":memory:")
        cls.embedder = KeywordEmbedder()
        cls.metadata = build_index(
            sample_chunks(),
            collection_name="test_credit_policy",
            embedder=cls.embedder,
            client=cls.client,
        )

    @classmethod
    def tearDownClass(cls):
        cls.client.close()

    def test_local_embedding_model_dimension_and_normalization(self):
        embedder = LocalSentenceEmbedder()
        vectors = embedder.encode(["mortgage debt-to-income", "mortgage LTV"])
        self.assertEqual(vectors.shape, (2, EMBEDDING_DIMENSION))
        np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)

    def test_qdrant_collection_is_created_and_readable(self):
        self.assertEqual(self.metadata["storage_mode"], "embedded_local")
        self.assertEqual(self.metadata["point_count"], 3)
        self.assertEqual(self.client.count("test_credit_policy", exact=True).count, 3)

    def test_retrieval_returns_expected_source_metadata(self):
        payload = retrieve_policy_context(
            "debt DTI obligations",
            k=2,
            collection_name="test_credit_policy",
            embedder=self.embedder,
            client=self.client,
        )
        self.assertEqual(payload["results"][0]["source_id"], "dti")
        required = {
            "text",
            "score",
            "source_title",
            "institution",
            "source_url",
            "page_or_section",
            "chunk_id",
        }
        self.assertTrue(required.issubset(payload["results"][0]))
        json.dumps(payload, allow_nan=False)

    def test_evaluation_separates_directed_and_out_of_domain_behavior(self):
        evaluation_set = [
            {
                "query_id": "directed_dti",
                "evaluation_group": "directed",
                "query_type": "directed",
                "query": "debt DTI obligations",
                "expected_source_ids": ["dti"],
                "expected_concepts": ["debt"],
            },
            {
                "query_id": "ood",
                "evaluation_group": "additional",
                "query_type": "out_of_domain",
                "query": "business checking account documents",
                "expected_no_relevant_source": True,
                "expected_source_ids": [],
                "expected_concepts": [],
            },
        ]
        result = evaluar_retrieval(
            evaluation_set,
            k=2,
            collection_name="test_credit_policy",
            embedder=self.embedder,
            client=self.client,
        )
        self.assertEqual(result["metrics"]["directed"]["source_hit_at_2"], 1.0)
        self.assertEqual(result["metrics"]["out_of_domain"]["query_count"], 1)
        self.assertEqual(
            result["metrics"]["out_of_domain"]["returned_results_count"], 1
        )
        self.assertEqual(result["metrics"]["out_of_domain"]["rejected_count"], 0)
        ood = next(row for row in result["queries"] if row["query_id"] == "ood")
        self.assertIsNone(ood["source_hit_at_k"])
        self.assertFalse(ood["rejection_applied"])


class TestXaiQueryHelper(unittest.TestCase):
    def test_queries_are_deterministic_and_json_serializable(self):
        evidence = {
            "top_positive_factors": [
                {"model_feature": "dti_category", "source_features": ["debt_to_income_ratio"]},
                {"model_feature": "loan_type", "source_features": ["loan_type"]},
            ],
            "top_negative_factors": [
                {"model_feature": "loan_to_property_value", "source_features": ["loan_amount", "property_value"]}
            ],
        }
        first = construir_queries_desde_evidence(evidence)
        second = construir_queries_desde_evidence(evidence)
        self.assertEqual(first, second)
        self.assertEqual(len(first["queries"]), 3)
        json.dumps(first, allow_nan=False)

    def test_audit_only_factors_are_structurally_excluded(self):
        evidence = {
            "top_positive_factors": [
                {
                    "model_feature": AUDIT_ONLY_COLUMNS[0],
                    "source_features": [AUDIT_ONLY_COLUMNS[0]],
                },
                {"model_feature": "income", "source_features": ["income"]},
            ],
            "top_negative_factors": [],
        }
        payload = construir_queries_desde_evidence(evidence)
        serialized = json.dumps(payload)
        for column in AUDIT_ONLY_COLUMNS:
            self.assertNotIn(column, serialized)
        self.assertEqual(payload["queries"][0]["origin_feature"], "income")


if __name__ == "__main__":
    unittest.main()
