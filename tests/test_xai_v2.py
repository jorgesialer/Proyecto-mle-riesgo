import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import mlflow
import numpy as np
import pandas as pd

from src.construir_dataset_v2 import AUDIT_ONLY_COLUMNS, MODEL_PREDICTOR_COLUMNS
from src.mlflow_utils import configurar_mlflow
from src.xai_v2 import (
    FeatureDescriptor,
    _global_importance_tables,
    cargar_champion,
    cargar_resumen,
    construir_descriptores_features,
    explicar_solicitud,
    generar_artifacts_xai,
    localizar_clase_positiva,
    obtener_etiqueta_hmda,
    registrar_artifacts_xai,
)


ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "data" / "hmda_2023_loan_approval_v2.csv"


class PositiveClassTests(unittest.TestCase):
    def test_positive_class_is_located_from_classes(self):
        estimator = type("Estimator", (), {"classes_": np.array([1, 0])})()
        self.assertEqual(localizar_clase_positiva(estimator), 0)

    def test_missing_positive_class_is_rejected(self):
        estimator = type("Estimator", (), {"classes_": np.array([0, 2])})()
        with self.assertRaisesRegex(ValueError, "clase positiva"):
            localizar_clase_positiva(estimator)


class GlobalAggregationTests(unittest.TestCase):
    def test_shap_is_summed_per_row_before_mean_absolute_value(self):
        descriptors = [
            FeatureDescriptor(0, "loan_type_1", "loan_type", "Loan type", ("loan_type",), "1", "Conventional"),
            FeatureDescriptor(1, "loan_type_2", "loan_type", "Loan type", ("loan_type",), "2", "FHA"),
            FeatureDescriptor(2, "income", "income", "Annual income", ("income",)),
        ]
        values = np.array([[1.0, -1.0, 0.5], [2.0, -2.0, 0.5]])
        grouped, transformed = _global_importance_tables(values, descriptors)

        grouped_by_name = grouped.set_index("model_feature")
        self.assertEqual(grouped.iloc[0]["model_feature"], "income")
        self.assertAlmostEqual(grouped_by_name.loc["loan_type", "mean_abs_shap"], 0.0)
        old_incorrect_value = transformed.loc[
            transformed["model_feature"] == "loan_type", "mean_abs_shap"
        ].sum()
        self.assertAlmostEqual(old_incorrect_value, 3.0)
        self.assertGreater(old_incorrect_value, grouped_by_name.loc["income", "mean_abs_shap"])

    def test_official_hmda_code_labels_and_unknown_fallback(self):
        self.assertEqual(obtener_etiqueta_hmda("loan_type", "3"), "VA")
        self.assertEqual(
            obtener_etiqueta_hmda("loan_purpose", 32), "Cash-out refinancing"
        )
        self.assertEqual(obtener_etiqueta_hmda("lien_status", "2"), "Subordinate lien")
        self.assertEqual(
            obtener_etiqueta_hmda("occupancy_type", "1"), "Principal residence"
        )
        self.assertEqual(
            obtener_etiqueta_hmda("construction_method", "2"), "Manufactured home"
        )
        self.assertEqual(obtener_etiqueta_hmda("loan_type", "999"), "unknown")


class ChampionXAITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipeline = cargar_champion()
        cls.summary = cargar_resumen()
        cls.data = pd.read_csv(DATASET_PATH, nrows=200)
        cls.raw = cls.data.loc[:, MODEL_PREDICTOR_COLUMNS].iloc[[0]]
        cls.evidence = explicar_solicitud(
            cls.pipeline,
            cls.raw,
            top_n=3,
            summary=cls.summary,
        )

    def test_champion_loads_with_expected_positive_class(self):
        self.assertEqual(
            localizar_clase_positiva(self.pipeline.named_steps["estimador"]), 1
        )

    def test_evidence_package_contract_and_signs(self):
        self.assertIn(self.evidence["prediction"], {0, 1})
        self.assertGreaterEqual(self.evidence["probability"], 0.0)
        self.assertLessEqual(self.evidence["probability"], 1.0)
        positive_index = localizar_clase_positiva(self.pipeline.named_steps["estimador"])
        expected_probability = self.pipeline.predict_proba(self.raw)[0, positive_index]
        self.assertAlmostEqual(self.evidence["probability"], expected_probability)
        self.assertTrue(
            all(item["shap_value"] > 0 for item in self.evidence["top_positive_factors"])
        )
        self.assertTrue(
            all(item["shap_value"] < 0 for item in self.evidence["top_negative_factors"])
        )

    def test_audit_only_never_appear_in_xai(self):
        serialized = json.dumps(self.evidence)
        for column in AUDIT_ONLY_COLUMNS:
            self.assertNotIn(column, serialized)
        descriptors = construir_descriptores_features(self.pipeline)
        for descriptor in descriptors:
            self.assertTrue(set(descriptor.source_features).isdisjoint(AUDIT_ONLY_COLUMNS))

    def test_feature_names_preserve_one_hot_context(self):
        descriptors = construir_descriptores_features(self.pipeline)
        self.assertEqual(len(descriptors), 45)
        categorical = [item for item in descriptors if item.category is not None]
        self.assertTrue(categorical)
        self.assertTrue(all(item.category is not None for item in categorical))
        self.assertEqual(len({item.transformed_feature for item in descriptors}), 45)
        loan_type_va = next(
            item
            for item in descriptors
            if item.model_feature == "loan_type" and str(item.category) == "3"
        )
        self.assertEqual(loan_type_va.category_label, "VA")
        self.assertEqual(loan_type_va.display_name, "Loan type = VA")

    def test_reconstructed_feature_names_must_match_sklearn_positionally(self):
        preprocessor = self.pipeline.named_steps["preprocesamiento"]
        changed = preprocessor.get_feature_names_out().copy()
        changed[0] = "unexpected_name"
        with patch.object(preprocessor, "get_feature_names_out", return_value=changed):
            with self.assertRaisesRegex(AssertionError, "posicion 0"):
                construir_descriptores_features(self.pipeline)

    def test_evidence_is_json_serializable_and_reproducible(self):
        encoded = json.dumps(self.evidence, allow_nan=False)
        self.assertTrue(encoded)
        repeated = explicar_solicitud(
            self.pipeline,
            self.raw,
            top_n=3,
            summary=self.summary,
        )
        self.assertEqual(self.evidence, repeated)
        reordered = explicar_solicitud(
            self.pipeline,
            self.raw.loc[:, list(reversed(MODEL_PREDICTOR_COLUMNS))],
            top_n=3,
            summary=self.summary,
        )
        self.assertEqual(self.evidence, reordered)
        unknown_summary = explicar_solicitud(
            self.pipeline,
            self.raw,
            top_n=3,
            summary={},
        )
        self.assertIsNone(unknown_summary["model"]["selected_configuration"])
        categorical_factors = [
            factor
            for factor in (
                unknown_summary["top_positive_factors"]
                + unknown_summary["top_negative_factors"]
            )
            if factor["raw_code"] is not None
        ]
        self.assertTrue(categorical_factors)
        self.assertTrue(all("category_label" in factor for factor in categorical_factors))
        json.dumps(unknown_summary, allow_nan=False)

    def test_artifact_generation_basic(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "first"
            repeated_output = Path(directory) / "second"
            result = generar_artifacts_xai(
                self.pipeline,
                self.data,
                output,
                sample_size=20,
                top_n=5,
                summary=self.summary,
            )
            expected = {
                "global_feature_importance.csv",
                "global_feature_importance.json",
                "global_feature_importance_transformed.csv",
                "global_feature_importance_transformed.json",
                "shap_summary.png",
                "shap_bar.png",
                "local_example.json",
                "metadata.json",
            }
            self.assertTrue(all((output / name).is_file() for name in expected))
            self.assertEqual(result["metadata"]["sample_size"], 20)
            self.assertEqual(len(result["top_features"]), 5)
            generar_artifacts_xai(
                self.pipeline,
                self.data,
                repeated_output,
                sample_size=20,
                top_n=5,
                summary=self.summary,
            )
            self.assertEqual(
                (output / "global_feature_importance.json").read_bytes(),
                (repeated_output / "global_feature_importance.json").read_bytes(),
            )


class XAITrackingTests(unittest.TestCase):
    def test_xai_artifacts_are_logged_in_one_descriptive_local_run(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            root = Path(directory)
            output = root / "xai"
            output.mkdir()
            (output / "metadata.json").write_text("{}", encoding="utf-8")
            environment = {
                "MLFLOW_BACKEND": "local",
                "MLFLOW_LOCAL_DB": str(root / "test.db"),
                "MLFLOW_LOCAL_ARTIFACT_ROOT": str(root / "mlartifacts"),
            }
            metadata = {
                "model_name": "credit-approval-v2",
                "model_version": "1",
                "method": "Tree SHAP",
                "sample_size": 20,
                "top_n": 5,
                "random_state": 42,
                "positive_class": 1,
                "dataset_version": "hmda_2023_loan_approval_v2",
                "dataset_sha256": "test-checksum",
                "global_aggregation": "sum per row then mean absolute value",
            }
            with patch.dict(os.environ, environment, clear=False):
                tracking = configurar_mlflow(
                    root, experiment_name="test-credit-approval-v2-xai"
                )
                result = registrar_artifacts_xai(output, tracking, metadata)
                run = mlflow.MlflowClient().get_run(result["run_id"])
                artifacts = mlflow.MlflowClient().list_artifacts(
                    result["run_id"], "xai"
                )
            self.assertEqual(run.data.tags["mlflow.runName"], "xai_catboost_champion_v2")
            self.assertEqual(run.data.tags["stage"], "xai")
            self.assertTrue(any(item.path.endswith("metadata.json") for item in artifacts))


if __name__ == "__main__":
    unittest.main()
