import os
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import mlflow
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.construir_dataset_v2 import AUDIT_ONLY_COLUMNS, TARGET_COLUMN
from src.entrenamiento_v2 import (
    POSITIVE_CLASS,
    RunSpec,
    construir_run_specs,
    crear_estimador,
    seleccionar_champion,
    serializar_pipeline,
)
from src.mlflow_utils import configurar_mlflow, validar_run_name
from src.preprocesamiento_v2 import crear_pipeline_v2, preparar_xy_v2


ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "data" / "hmda_2023_loan_approval_v2.csv"


class ModelConfigurationV2Tests(unittest.TestCase):
    def test_only_three_approved_model_families_are_configured(self):
        specs = construir_run_specs()
        self.assertEqual(len(specs), 11)
        self.assertEqual(
            {spec.model_family for spec in specs},
            {"random_forest", "xgboost", "catboost"},
        )
        self.assertEqual(len({spec.run_name for spec in specs}), len(specs))
        for spec in specs:
            validar_run_name(spec.run_name)

    def test_model_factory_returns_expected_classes(self):
        specs = {spec.model_family: spec for spec in construir_run_specs()}
        self.assertIsInstance(crear_estimador(specs["random_forest"]), RandomForestClassifier)
        self.assertIsInstance(crear_estimador(specs["xgboost"]), XGBClassifier)
        self.assertIsInstance(crear_estimador(specs["catboost"]), CatBoostClassifier)

    def test_positive_class_is_loan_approved_one(self):
        self.assertEqual(POSITIVE_CLASS, 1)

    def test_champion_selection_ignores_holdout_fields(self):
        weak_cv = {
            "run_name": "rf_candidate_full",
            "metrics": {
                "cv_f1_mean": 0.80,
                "cv_average_precision_mean": 0.90,
                "cv_roc_auc_mean": 0.88,
                "holdout_f1": 0.99,
            },
        }
        strong_cv = {
            "run_name": "xgb_candidate_full",
            "metrics": {
                "cv_f1_mean": 0.85,
                "cv_average_precision_mean": 0.91,
                "cv_roc_auc_mean": 0.89,
                "holdout_f1": 0.01,
            },
        }
        self.assertIs(seleccionar_champion([weak_cv, strong_cv]), strong_cv)


class MlflowV2Tests(unittest.TestCase):
    def test_local_mlflow_logging_with_descriptive_name(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            root = Path(directory)
            environment = {
                "MLFLOW_BACKEND": "local",
                "MLFLOW_LOCAL_DB": str(root / "test.db"),
                "MLFLOW_LOCAL_ARTIFACT_ROOT": str(root / "artifacts"),
            }
            with patch.dict(os.environ, environment, clear=False):
                config = configurar_mlflow(
                    root,
                    experiment_name="test-credit-approval-v2",
                )
                with mlflow.start_run(
                    experiment_id=config.experiment_id,
                    run_name="rf_test_logging",
                ) as run:
                    mlflow.log_param("model_family", "random_forest")
                    mlflow.log_metric("cv_f1_mean", 0.8)
                stored = mlflow.MlflowClient().get_run(run.info.run_id)
                self.assertEqual(stored.data.tags["mlflow.runName"], "rf_test_logging")
                self.assertEqual(stored.data.params["model_family"], "random_forest")
                self.assertEqual(stored.data.metrics["cv_f1_mean"], 0.8)

    def test_dagshub_backend_requires_explicit_credentials(self):
        cleared = {
            "MLFLOW_TRACKING_URI": "",
            "MLFLOW_TRACKING_USERNAME": "",
            "MLFLOW_TRACKING_PASSWORD": "",
        }
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ, cleared, clear=False
        ):
            with self.assertRaisesRegex(EnvironmentError, "faltan variables"):
                configurar_mlflow(Path(directory), backend="dagshub")


class ChampionSerializationV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = pd.read_csv(DATASET_PATH).iloc[:500].copy()

    def test_audit_only_never_enter_serialized_pipeline(self):
        X, y = preparar_xy_v2(self.data)
        self.assertTrue(set(X.columns).isdisjoint(AUDIT_ONLY_COLUMNS))
        self.assertNotIn(TARGET_COLUMN, X.columns)
        pipeline = crear_pipeline_v2(
            RandomForestClassifier(n_estimators=5, random_state=42, n_jobs=1)
        )
        pipeline.fit(X, y)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "champion.pkl"
            serializar_pipeline(pipeline, path)
            with path.open("rb") as file:
                restored = pickle.load(file)
            self.assertEqual(restored.predict(X.iloc[:3]).shape, (3,))
            self.assertEqual(restored.predict_proba(X.iloc[:3]).shape, (3, 2))


if __name__ == "__main__":
    unittest.main()
