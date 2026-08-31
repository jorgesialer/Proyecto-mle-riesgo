"""Feature engineering y preprocesamiento reproducible para HMDA V2.

Este modulo define el contrato previo al entrenamiento. No selecciona ni
entrena un modelo final; el estimador se inyecta explicitamente.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.construir_dataset_v2 import (
    AUDIT_ONLY_COLUMNS,
    MODEL_PREDICTOR_COLUMNS,
    TARGET_COLUMN,
    obtener_predictores_entrenamiento,
)


RANDOM_STATE = 42
TEST_SIZE = 0.20

ENGINEERED_NUMERIC_COLUMNS = (
    "income",
    "loan_amount",
    "combined_loan_to_value_ratio",
    "property_value",
    "loan_term_years",
    "loan_to_income",
    "property_value_to_income",
    "loan_to_property_value",
    "non_amortizing_feature_count",
)

ENGINEERED_CATEGORICAL_COLUMNS = (
    "loan_purpose",
    "loan_type",
    "lien_status",
    "preapproval",
    "dti_category",
    "occupancy_type",
    "construction_method",
    "total_units",
    "submission_of_application",
    "interest_only_payment",
    "balloon_payment",
)

ENGINEERED_MODEL_COLUMNS = (
    ENGINEERED_NUMERIC_COLUMNS + ENGINEERED_CATEGORICAL_COLUMNS
)

FEATURE_CONFIG_FULL = "full"
FEATURE_CONFIG_WITHOUT_WORKFLOW = "without_workflow_features"
WORKFLOW_RAW_COLUMNS = (
    "income",
    "debt_to_income_ratio",
    "combined_loan_to_value_ratio",
    "property_value",
)
WORKFLOW_ENGINEERED_COLUMNS = frozenset(
    {
        "income",
        "combined_loan_to_value_ratio",
        "property_value",
        "loan_to_income",
        "property_value_to_income",
        "loan_to_property_value",
        "dti_category",
    }
)

DTI_SOURCE_BANDS = {
    "<20%": "lt_20",
    "20%-<30%": "20_to_29",
    "30%-<36%": "30_to_35",
    "50%-60%": "50_to_60",
    ">60%": "gt_60",
}


def validar_columnas_predictoras_v2(columnas: Any) -> None:
    """Exige exactamente las 16 variables crudas autorizadas para V2."""
    received = list(columnas)
    if len(received) != len(set(received)):
        raise ValueError("Contrato V2 invalido: hay columnas duplicadas")
    expected = set(MODEL_PREDICTOR_COLUMNS)
    actual = set(received)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    errors = []
    if missing:
        errors.append(f"faltan columnas: {missing}")
    if unexpected:
        errors.append(f"columnas inesperadas: {unexpected}")
    if errors:
        raise ValueError("Contrato V2 invalido: " + "; ".join(errors))


def categorizar_dti(value: Any) -> str | float:
    """Normaliza las bandas HMDA y agrupa los valores numericos publicados."""
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if text in DTI_SOURCE_BANDS:
        return DTI_SOURCE_BANDS[text]
    try:
        numeric = float(text)
    except ValueError:
        return np.nan
    if numeric < 20:
        return "lt_20"
    if numeric < 30:
        return "20_to_29"
    if numeric < 36:
        return "30_to_35"
    if numeric < 43:
        return "36_to_42"
    if numeric < 50:
        return "43_to_49"
    if numeric <= 60:
        return "50_to_60"
    return "gt_60"


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    valid_denominator = denominator.where(denominator > 0)
    result = numerator / valid_denominator
    return result.replace([np.inf, -np.inf], np.nan)


def _as_categorical(series: pd.Series) -> pd.Series:
    return series.map(lambda value: str(value) if pd.notna(value) else np.nan)


class HMDAFeatureEngineer(TransformerMixin, BaseEstimator):
    """Transformacion determinista de predictores HMDA crudos."""

    def fit(self, X: pd.DataFrame, y: Any = None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("HMDAFeatureEngineer requiere un pandas DataFrame")
        validar_columnas_predictoras_v2(X.columns)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = len(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("HMDAFeatureEngineer requiere un pandas DataFrame")
        validar_columnas_predictoras_v2(X.columns)

        income = pd.to_numeric(X["income"], errors="coerce")
        annual_income = income * 1_000.0
        loan_amount = pd.to_numeric(X["loan_amount"], errors="coerce")
        property_value = pd.to_numeric(X["property_value"], errors="coerce")
        loan_term = pd.to_numeric(X["loan_term"], errors="coerce")

        result = pd.DataFrame(index=X.index)
        result["income"] = income
        result["loan_amount"] = loan_amount
        result["combined_loan_to_value_ratio"] = pd.to_numeric(
            X["combined_loan_to_value_ratio"], errors="coerce"
        )
        result["property_value"] = property_value
        result["loan_term_years"] = loan_term / 12.0
        result["loan_to_income"] = _safe_ratio(loan_amount, annual_income)
        result["property_value_to_income"] = _safe_ratio(
            property_value, annual_income
        )
        result["loan_to_property_value"] = _safe_ratio(
            loan_amount, property_value
        )

        contract_flags = pd.DataFrame(
            {
                column: pd.to_numeric(X[column], errors="coerce").map({1: 1, 2: 0})
                for column in ("interest_only_payment", "balloon_payment")
            },
            index=X.index,
        )
        result["non_amortizing_feature_count"] = contract_flags.sum(
            axis=1, min_count=len(contract_flags.columns)
        )

        for column in (
            "loan_purpose",
            "loan_type",
            "lien_status",
            "preapproval",
            "occupancy_type",
            "construction_method",
            "total_units",
            "submission_of_application",
            "interest_only_payment",
            "balloon_payment",
        ):
            result[column] = _as_categorical(X[column])
        result["dti_category"] = X["debt_to_income_ratio"].map(categorizar_dti)

        return result.loc[:, ENGINEERED_MODEL_COLUMNS]

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.asarray(ENGINEERED_MODEL_COLUMNS, dtype=object)


def obtener_columnas_modelo_v2(
    feature_config: str = FEATURE_CONFIG_FULL,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Devuelve columnas efectivamente usadas por cada configuracion."""
    if feature_config == FEATURE_CONFIG_FULL:
        return ENGINEERED_NUMERIC_COLUMNS, ENGINEERED_CATEGORICAL_COLUMNS
    if feature_config == FEATURE_CONFIG_WITHOUT_WORKFLOW:
        numeric = tuple(
            column
            for column in ENGINEERED_NUMERIC_COLUMNS
            if column not in WORKFLOW_ENGINEERED_COLUMNS
        )
        categorical = tuple(
            column
            for column in ENGINEERED_CATEGORICAL_COLUMNS
            if column not in WORKFLOW_ENGINEERED_COLUMNS
        )
        return numeric, categorical
    raise ValueError(f"Configuracion de features V2 desconocida: {feature_config!r}")


def crear_preprocesador_v2(
    feature_config: str = FEATURE_CONFIG_FULL,
) -> ColumnTransformer:
    """Crea imputacion y encoding que se ajustaran solo con training data."""
    numeric_columns, categorical_columns = obtener_columnas_modelo_v2(feature_config)
    numeric = Pipeline(steps=[("imputar", SimpleImputer(strategy="median"))])
    categorical = Pipeline(
        steps=[
            ("imputar", SimpleImputer(strategy="most_frequent")),
            ("codificar", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numericas", numeric, list(numeric_columns)),
            ("categoricas", categorical, list(categorical_columns)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def crear_pipeline_v2(
    estimator: Any,
    feature_config: str = FEATURE_CONFIG_FULL,
) -> Pipeline:
    """Compone feature engineering, preprocesamiento y un estimador inyectado."""
    if estimator is None or not hasattr(estimator, "fit") or not hasattr(estimator, "predict"):
        raise TypeError("Debe proporcionarse un estimador sklearn con fit y predict")
    return Pipeline(
        steps=[
            ("feature_engineering", HMDAFeatureEngineer()),
            ("preprocesamiento", crear_preprocesador_v2(feature_config)),
            ("estimador", estimator),
        ]
    )


def preparar_xy_v2(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separa target y audit-only mediante una whitelist estructural."""
    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"Falta el target {TARGET_COLUMN}")
    target = data[TARGET_COLUMN]
    if target.isna().any() or not set(target.unique()).issubset({0, 1}):
        raise ValueError(f"{TARGET_COLUMN} debe ser binario y no contener nulos")
    predictors = obtener_predictores_entrenamiento(data)
    forbidden = set(predictors.columns) & (set(AUDIT_ONLY_COLUMNS) | {TARGET_COLUMN})
    if forbidden:
        raise AssertionError(f"Leakage estructural en X: {sorted(forbidden)}")
    return predictors, target.astype("int8").copy()


def separar_train_test_v2(
    data: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Reserva un holdout estratificado antes de ajustar transformaciones."""
    X, y = preparar_xy_v2(data)
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
