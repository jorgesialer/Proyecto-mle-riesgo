import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from src.construir_dataset_v2 import (
    AUDIT_ONLY_COLUMNS,
    MODEL_PREDICTOR_COLUMNS,
    TARGET_COLUMN,
)
from src.preprocesamiento_v2 import (
    ENGINEERED_MODEL_COLUMNS,
    FEATURE_CONFIG_WITHOUT_WORKFLOW,
    HMDAFeatureEngineer,
    WORKFLOW_ENGINEERED_COLUMNS,
    categorizar_dti,
    crear_pipeline_v2,
    obtener_columnas_modelo_v2,
    preparar_xy_v2,
    separar_train_test_v2,
)


ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "data" / "hmda_2023_loan_approval_v2.csv"


def predictor_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "income": 100.0,
                "loan_amount": 250_000.0,
                "loan_term": 360.0,
                "loan_purpose": 1,
                "loan_type": 1,
                "lien_status": 1,
                "preapproval": 2,
                "debt_to_income_ratio": "30%-<36%",
                "combined_loan_to_value_ratio": 80.0,
                "property_value": 310_000.0,
                "occupancy_type": 1,
                "construction_method": 1,
                "total_units": 1,
                "submission_of_application": 1,
                "interest_only_payment": 1,
                "balloon_payment": 2,
            },
            {
                "income": np.nan,
                "loan_amount": 180_000.0,
                "loan_term": 180.0,
                "loan_purpose": 2,
                "loan_type": 2,
                "lien_status": 1,
                "preapproval": 2,
                "debt_to_income_ratio": "42",
                "combined_loan_to_value_ratio": np.nan,
                "property_value": 220_000.0,
                "occupancy_type": 1,
                "construction_method": 1,
                "total_units": 2,
                "submission_of_application": 2,
                "interest_only_payment": 2,
                "balloon_payment": 2,
            },
        ],
        columns=MODEL_PREDICTOR_COLUMNS,
    )


class FeatureEngineeringV2Tests(unittest.TestCase):
    def test_feature_engineering_is_reproducible_and_interpretable(self):
        X = predictor_frame()
        transformer = HMDAFeatureEngineer().fit(X)
        first = transformer.transform(X)
        second = transformer.transform(X.copy())
        assert_frame_equal(first, second)
        self.assertEqual(tuple(first.columns), ENGINEERED_MODEL_COLUMNS)
        self.assertAlmostEqual(first.loc[0, "loan_to_income"], 2.5)
        self.assertAlmostEqual(first.loc[0, "property_value_to_income"], 3.1)
        self.assertAlmostEqual(
            first.loc[0, "loan_to_property_value"], 250_000 / 310_000
        )
        self.assertEqual(first.loc[0, "loan_term_years"], 30.0)
        self.assertEqual(first.loc[0, "dti_category"], "30_to_35")
        self.assertEqual(first.loc[0, "non_amortizing_feature_count"], 1)

    def test_dti_normalization_uses_documented_hmda_bands(self):
        self.assertEqual(categorizar_dti("<20%"), "lt_20")
        self.assertEqual(categorizar_dti("42"), "36_to_42")
        self.assertEqual(categorizar_dti(49), "43_to_49")
        self.assertEqual(categorizar_dti(">60%"), "gt_60")
        self.assertTrue(pd.isna(categorizar_dti(np.nan)))

    def test_nonpositive_denominators_do_not_create_infinite_ratios(self):
        X = predictor_frame().iloc[[0]].copy()
        X.loc[:, "income"] = 0
        X.loc[:, "property_value"] = 0
        engineered = HMDAFeatureEngineer().fit_transform(X)
        for column in (
            "loan_to_income",
            "property_value_to_income",
            "loan_to_property_value",
        ):
            self.assertTrue(pd.isna(engineered.iloc[0][column]))

    def test_feature_engineer_rejects_audit_or_unexpected_columns(self):
        X = predictor_frame()
        X["derived_race"] = "White"
        with self.assertRaisesRegex(ValueError, "columnas inesperadas"):
            HMDAFeatureEngineer().fit(X)


class PipelineV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = pd.read_csv(DATASET_PATH)

    def test_audit_only_and_target_are_structurally_excluded_from_x(self):
        X, y = preparar_xy_v2(self.data)
        self.assertEqual(tuple(X.columns), MODEL_PREDICTOR_COLUMNS)
        self.assertTrue(set(X.columns).isdisjoint(AUDIT_ONLY_COLUMNS))
        self.assertNotIn(TARGET_COLUMN, X.columns)
        self.assertEqual(y.name, TARGET_COLUMN)

    def test_train_test_split_is_reproducible_and_stratified(self):
        first = separar_train_test_v2(self.data)
        second = separar_train_test_v2(self.data)
        for left, right in zip(first, second):
            self.assertTrue(left.index.equals(right.index))
        _, _, y_train, y_test = first
        self.assertAlmostEqual(y_train.mean(), y_test.mean(), places=3)

    def test_pipeline_fit_predict_and_unknown_category(self):
        subset = self.data.iloc[:1_000].copy()
        X, y = preparar_xy_v2(subset)
        pipeline = crear_pipeline_v2(DummyClassifier(strategy="prior"))
        self.assertIsInstance(pipeline, Pipeline)
        pipeline.fit(X, y)
        unknown = X.iloc[:3].copy()
        unknown.loc[:, "loan_purpose"] = 999
        self.assertEqual(pipeline.predict(unknown).shape, (3,))
        self.assertEqual(pipeline.predict_proba(unknown).shape, (3, 2))

    def test_imputation_and_encoding_live_inside_pipeline(self):
        pipeline = crear_pipeline_v2(DummyClassifier())
        steps = pipeline.named_steps
        self.assertIn("feature_engineering", steps)
        self.assertIn("preprocesamiento", steps)
        transformers = {
            name: transformer
            for name, transformer, _ in steps["preprocesamiento"].transformers
        }
        self.assertEqual(
            transformers["numericas"].named_steps["imputar"].strategy, "median"
        )
        categorical = transformers["categoricas"].named_steps
        self.assertEqual(categorical["imputar"].strategy, "most_frequent")
        self.assertEqual(categorical["codificar"].handle_unknown, "ignore")

    def test_robustness_config_excludes_workflow_and_dependent_features(self):
        numeric, categorical = obtener_columnas_modelo_v2(
            FEATURE_CONFIG_WITHOUT_WORKFLOW
        )
        self.assertTrue(
            set(numeric + categorical).isdisjoint(WORKFLOW_ENGINEERED_COLUMNS)
        )


if __name__ == "__main__":
    unittest.main()
