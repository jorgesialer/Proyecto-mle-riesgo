import json
import unittest
from pathlib import Path

import pandas as pd

from src.construir_dataset_v2 import (
    AUDIT_ONLY_COLUMNS,
    FINAL_COLUMNS,
    LEAKAGE_BLACKLIST,
    MODEL_PREDICTOR_COLUMNS,
    SOURCE_TO_FINAL,
    TARGET_COLUMN,
    mapear_target,
    muestrear_registros,
    obtener_predictores_entrenamiento,
    validar_dataset,
)


ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "data" / "hmda_2023_loan_approval_v2.csv"
METADATA_PATH = ROOT / "data" / "hmda_2023_loan_approval_v2.metadata.json"


def raw_row(index: int, action_taken: str = "1", **overrides: str) -> dict[str, str]:
    row = {source: "1" for source in SOURCE_TO_FINAL}
    row.update(
        {
            "activity_year": "2023",
            "action_taken": action_taken,
            "reverse_mortgage": "2",
            "open-end_line_of_credit": "2",
            "business_or_commercial_purpose": "2",
            "total_units": "1",
            "income": str(index),
            "loan_amount": "250000",
            "loan_term": "360",
            "loan_to_value_ratio": "80",
            "property_value": "310000",
            "debt_to_income_ratio": "30%-<36%",
            "applicant_age": "35-44",
            "derived_race": "White",
            "derived_ethnicity": "Not Hispanic or Latino",
            "derived_sex": "Male",
        }
    )
    row.update(overrides)
    return row


class DatasetV2UnitTests(unittest.TestCase):
    def test_target_mapping(self):
        self.assertEqual(mapear_target("1"), 1)
        self.assertEqual(mapear_target("2"), 1)
        self.assertEqual(mapear_target("3"), 0)
        with self.assertRaisesRegex(ValueError, "fuera del universo binario"):
            mapear_target("4")

    def test_reservoir_sampling_is_reproducible(self):
        rows = [raw_row(i, "3" if i % 3 == 0 else "1") for i in range(100)]
        first, first_stats = muestrear_registros(rows, sample_size=20, random_state=42)
        second, second_stats = muestrear_registros(rows, sample_size=20, random_state=42)
        self.assertEqual(first, second)
        self.assertEqual(first_stats, second_stats)
        self.assertEqual(len(first), 20)

    def test_population_filters_are_applied_before_sampling(self):
        rows = [
            raw_row(1, "1"),
            raw_row(2, "3"),
            raw_row(3, "1", reverse_mortgage="1"),
            raw_row(4, "1", **{"open-end_line_of_credit": "1"}),
            raw_row(5, "1", business_or_commercial_purpose="1"),
        ]
        sample, stats = muestrear_registros(rows, sample_size=2, random_state=42)
        self.assertEqual(len(sample), 2)
        self.assertEqual(stats["eligible_population_rows"], 2)

    def test_predictor_missingness_does_not_remove_rows_before_sampling(self):
        row = raw_row(
            1,
            "3",
            income="NA",
            debt_to_income_ratio="Exempt",
            loan_to_value_ratio="NA",
            property_value="NA",
            submission_of_application="1111",
        )
        sample, stats = muestrear_registros([row], sample_size=1, random_state=42)
        self.assertEqual(stats["eligible_population_rows"], 1)
        for column in (
            "income",
            "debt_to_income_ratio",
            "combined_loan_to_value_ratio",
            "property_value",
            "submission_of_application",
        ):
            self.assertIsNone(sample[0][column])

    def test_audit_only_columns_cannot_enter_training_predictors(self):
        records, _ = muestrear_registros(
            [raw_row(1, "1"), raw_row(2, "3")], sample_size=2
        )
        data = pd.DataFrame.from_records(records, columns=FINAL_COLUMNS)
        predictors = obtener_predictores_entrenamiento(data)
        self.assertEqual(tuple(predictors.columns), MODEL_PREDICTOR_COLUMNS)
        self.assertTrue(set(predictors.columns).isdisjoint(AUDIT_ONLY_COLUMNS))
        self.assertNotIn(TARGET_COLUMN, predictors.columns)

    def test_validation_rejects_blacklisted_column(self):
        records, _ = muestrear_registros(
            [raw_row(1, "1"), raw_row(2, "3")], sample_size=2
        )
        data = pd.DataFrame.from_records(records, columns=FINAL_COLUMNS)
        data["action_taken"] = [1, 3]
        with self.assertRaisesRegex(ValueError, "Esquema Dataset V2 invalido"):
            validar_dataset(data, expected_rows=2)


class DatasetV2ArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not DATASET_PATH.exists() or not METADATA_PATH.exists():
            raise AssertionError(
                "Faltan los artefactos Dataset V2. Ejecute: "
                "python -m src.construir_dataset_v2"
            )
        cls.data = pd.read_csv(DATASET_PATH)
        cls.metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))

    def test_shape_and_schema(self):
        self.assertEqual(self.data.shape, (50_000, 21))
        self.assertEqual(tuple(self.data.columns), FINAL_COLUMNS)

    def test_low_information_features_are_absent(self):
        removed = {
            "negative_amortization",
            "introductory_rate_period",
            "other_nonamortizing_features",
        }
        self.assertTrue(removed.isdisjoint(self.data.columns))
        self.assertIn("total_units", self.data.columns)

    def test_target_is_binary_without_missing_values(self):
        self.assertFalse(self.data[TARGET_COLUMN].isna().any())
        self.assertEqual(set(self.data[TARGET_COLUMN].unique()), {0, 1})

    def test_action_taken_and_blacklist_are_absent(self):
        self.assertNotIn("action_taken", self.data.columns)
        self.assertTrue(set(self.data.columns).isdisjoint(LEAKAGE_BLACKLIST))

    def test_audit_only_columns_have_separate_role(self):
        roles = self.metadata["column_roles"]
        self.assertEqual(tuple(roles["audit_only"]), AUDIT_ONLY_COLUMNS)
        self.assertTrue(set(roles["predictor"]).isdisjoint(AUDIT_ONLY_COLUMNS))
        self.assertTrue(set(roles["predictor"]).isdisjoint({TARGET_COLUMN}))

    def test_column_names_are_unique(self):
        self.assertFalse(self.data.columns.duplicated().any())

    def test_file_is_smaller_than_100_mib(self):
        self.assertLess(DATASET_PATH.stat().st_size, 100 * 1024 * 1024)

    def test_metadata_records_reproducibility_contract(self):
        self.assertEqual(self.metadata["sampling"]["random_state"], 42)
        self.assertEqual(self.metadata["shape"], {"rows": 50_000, "columns": 21})
        self.assertFalse(self.metadata["source"]["raw_national_file_persisted"])


if __name__ == "__main__":
    unittest.main()
