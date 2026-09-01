"""Explicabilidad SHAP reusable para el pipeline champion CatBoost V2."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import shap
from catboost import CatBoostClassifier
from dotenv import load_dotenv

from src.construir_dataset_v2 import AUDIT_ONLY_COLUMNS, MODEL_PREDICTOR_COLUMNS
from src.entrenamiento_v2 import DATASET_VERSION, EXPERIMENT_NAME, REGISTERED_MODEL_NAME
from src.mlflow_utils import TrackingConfig, configurar_mlflow, validar_run_name
from src.preprocesamiento_v2 import (
    RANDOM_STATE,
    separar_train_test_v2,
    validar_columnas_predictoras_v2,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_PATH = ROOT / "data" / "hmda_2023_loan_approval_v2.csv"
DEFAULT_CHAMPION_PATH = ROOT / "artifacts" / "pipeline_champion_v2.pkl"
DEFAULT_SUMMARY_PATH = ROOT / "artifacts" / "benchmark_v2_summary.json"
DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "xai"
DEFAULT_GLOBAL_SAMPLE_SIZE = 1_000
DEFAULT_TOP_N = 10
POSITIVE_CLASS = 1
XAI_RUN_NAME = "xai_catboost_champion_v2"
NON_CAUSALITY_WARNING = (
    "SHAP describe contribuciones del modelo a esta prediccion; no demuestra "
    "causalidad, capacidad de repago ni una politica de credito normativa."
)

BUSINESS_LABELS = {
    "income": "Annual income (USD thousands)",
    "loan_amount": "Requested loan amount",
    "combined_loan_to_value_ratio": "Combined loan-to-value ratio",
    "property_value": "Property value",
    "loan_term_years": "Loan term (years)",
    "loan_to_income": "Loan-to-income ratio",
    "property_value_to_income": "Property-value-to-income ratio",
    "loan_to_property_value": "Loan-to-property-value ratio",
    "non_amortizing_feature_count": "Non-amortizing feature count",
    "loan_purpose": "Loan purpose",
    "loan_type": "Loan type",
    "lien_status": "Lien status",
    "preapproval": "Preapproval status",
    "dti_category": "Debt-to-income band",
    "occupancy_type": "Occupancy type",
    "construction_method": "Construction method",
    "total_units": "Property units",
    "submission_of_application": "Application submission channel",
    "interest_only_payment": "Interest-only payment flag",
    "balloon_payment": "Balloon payment flag",
}

HMDA_CATEGORY_LABELS = {
    "loan_purpose": {
        "1": "Home purchase",
        "2": "Home improvement",
        "31": "Refinancing",
        "32": "Cash-out refinancing",
        "4": "Other purpose",
        "5": "Not applicable",
    },
    "loan_type": {
        "1": "Conventional",
        "2": "FHA",
        "3": "VA",
        "4": "USDA/RHS/FSA",
    },
    "lien_status": {
        "1": "First lien",
        "2": "Subordinate lien",
    },
    "occupancy_type": {
        "1": "Principal residence",
        "2": "Second residence",
        "3": "Investment property",
    },
    "construction_method": {
        "1": "Site-built",
        "2": "Manufactured home",
    },
}

FEATURE_SOURCES = {
    "loan_term_years": ("loan_term",),
    "loan_to_income": ("loan_amount", "income"),
    "property_value_to_income": ("property_value", "income"),
    "loan_to_property_value": ("loan_amount", "property_value"),
    "dti_category": ("debt_to_income_ratio",),
    "non_amortizing_feature_count": (
        "interest_only_payment",
        "balloon_payment",
    ),
}


@dataclass(frozen=True)
class FeatureDescriptor:
    index: int
    transformed_feature: str
    model_feature: str
    business_feature: str
    source_features: tuple[str, ...]
    category: Any | None = None
    category_label: str | None = None

    @property
    def display_name(self) -> str:
        if self.category is None:
            return self.business_feature
        if self.category_label == "unknown":
            return f"{self.business_feature} = unknown (code {self.category})"
        return f"{self.business_feature} = {self.category_label or self.category}"


def _json_value(value: Any) -> Any:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _normalizar_codigo_hmda(value: Any) -> str:
    text = str(_json_value(value)).strip()
    try:
        numeric = float(text)
    except ValueError:
        return text
    return str(int(numeric)) if numeric.is_integer() else text


def obtener_etiqueta_hmda(feature: str, raw_code: Any) -> str | None:
    """Traduce solo enumeraciones oficiales; códigos desconocidos no se infieren."""
    mapping = HMDA_CATEGORY_LABELS.get(feature)
    if mapping is None:
        return None
    return mapping.get(_normalizar_codigo_hmda(raw_code), "unknown")


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def cargar_resumen(path: Path = DEFAULT_SUMMARY_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def cargar_champion(path: Path = DEFAULT_CHAMPION_PATH):
    """Carga y valida el pipeline champion local sin reentrenarlo."""
    if not path.exists():
        raise FileNotFoundError(f"No existe el champion V2: {path}")
    with path.open("rb") as file:
        pipeline = pickle.load(file)
    expected_steps = {"feature_engineering", "preprocesamiento", "estimador"}
    if not expected_steps.issubset(pipeline.named_steps):
        raise TypeError("El artefacto no cumple el contrato del pipeline V2")
    if not isinstance(pipeline.named_steps["estimador"], CatBoostClassifier):
        raise TypeError("El champion persistido no es CatBoostClassifier")
    localizar_clase_positiva(pipeline.named_steps["estimador"])
    return pipeline


def localizar_clase_positiva(estimator: Any, positive_class: int = POSITIVE_CLASS) -> int:
    """Localiza LoanApproved=1 sin asumir una posicion fija en predict_proba."""
    classes = np.asarray(getattr(estimator, "classes_", []))
    matches = np.flatnonzero(classes == positive_class)
    if len(matches) != 1:
        raise ValueError(
            f"El estimador debe contener exactamente una clase positiva {positive_class}; "
            f"classes_={classes.tolist()}"
        )
    return int(matches[0])


def construir_descriptores_features(pipeline: Any) -> list[FeatureDescriptor]:
    """Mapea la salida exacta del ColumnTransformer a conceptos de negocio."""
    preprocessor = pipeline.named_steps["preprocesamiento"]
    transformed_names = [str(name) for name in preprocessor.get_feature_names_out()]
    numeric_columns = list(preprocessor.transformers_[0][2])
    categorical_columns = list(preprocessor.transformers_[1][2])
    encoder = preprocessor.named_transformers_["categoricas"].named_steps["codificar"]
    reconstructed_names = numeric_columns + list(
        encoder.get_feature_names_out(categorical_columns)
    )
    reconstructed_names = [str(name) for name in reconstructed_names]
    if len(reconstructed_names) != len(transformed_names):
        raise AssertionError(
            "Contrato XAI roto: cantidad de nombres reconstruidos distinta de "
            "preprocessor.get_feature_names_out()"
        )
    for index, (reconstructed, actual) in enumerate(
        zip(reconstructed_names, transformed_names)
    ):
        if reconstructed != actual:
            raise AssertionError(
                "Contrato XAI roto en posicion "
                f"{index}: reconstruido={reconstructed!r}, sklearn={actual!r}"
            )

    descriptors: list[FeatureDescriptor] = []
    for column in numeric_columns:
        descriptors.append(
            FeatureDescriptor(
                index=len(descriptors),
                transformed_feature=reconstructed_names[len(descriptors)],
                model_feature=column,
                business_feature=BUSINESS_LABELS[column],
                source_features=FEATURE_SOURCES.get(column, (column,)),
            )
        )
    for column, categories in zip(categorical_columns, encoder.categories_):
        for category in categories:
            descriptors.append(
                FeatureDescriptor(
                    index=len(descriptors),
                    transformed_feature=reconstructed_names[len(descriptors)],
                    model_feature=column,
                    business_feature=BUSINESS_LABELS[column],
                    source_features=FEATURE_SOURCES.get(column, (column,)),
                    category=_json_value(category),
                    category_label=obtener_etiqueta_hmda(column, category),
                )
            )
    if len(descriptors) != len(transformed_names):
        raise ValueError("El mapping XAI no coincide con la salida del preprocesador")
    forbidden = set(AUDIT_ONLY_COLUMNS)
    if any(forbidden.intersection(item.source_features) for item in descriptors):
        raise AssertionError("Una feature audit-only alcanzo el mapping XAI")
    return descriptors


def _transformar_para_estimador(
    pipeline: Any, raw: pd.DataFrame
) -> tuple[pd.DataFrame, np.ndarray, list[FeatureDescriptor]]:
    validar_columnas_predictoras_v2(raw.columns)
    raw = raw.loc[:, MODEL_PREDICTOR_COLUMNS]
    engineered = pipeline.named_steps["feature_engineering"].transform(raw)
    transformed = pipeline.named_steps["preprocesamiento"].transform(engineered)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    matrix = np.asarray(transformed, dtype=float)
    descriptors = construir_descriptores_features(pipeline)
    if matrix.shape[1] != len(descriptors):
        raise ValueError("La matriz explicada y el mapping tienen distinta dimension")
    return engineered, matrix, descriptors


def _seleccionar_salida_positiva(
    values: Any,
    positive_index: int,
    n_classes: int,
    n_samples: int,
    n_features: int,
) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 2 and array.shape == (n_samples, n_features):
        if n_classes == 2 and positive_index == 0:
            return -array
        return array
    if array.ndim == 3 and array.shape == (n_samples, n_features, n_classes):
        return array[:, :, positive_index]
    if array.ndim == 3 and array.shape == (n_classes, n_samples, n_features):
        return array[positive_index]
    raise ValueError(f"Forma SHAP no soportada: {array.shape}")


def _seleccionar_base_value(
    base_values: Any, positive_index: int, n_classes: int, n_samples: int
) -> np.ndarray:
    array = np.asarray(base_values, dtype=float)
    if array.ndim == 0:
        selected = np.repeat(float(array), n_samples)
    elif array.ndim == 1 and len(array) == n_samples:
        selected = array
    elif array.ndim == 1 and len(array) == n_classes:
        selected = np.repeat(array[positive_index], n_samples)
    elif array.ndim == 2 and array.shape == (n_samples, n_classes):
        selected = array[:, positive_index]
    else:
        raise ValueError(f"Forma de base_value no soportada: {array.shape}")
    if n_classes == 2 and positive_index == 0 and array.shape in {(), (n_samples,)}:
        return -selected
    return selected


def calcular_shap(pipeline: Any, matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Calcula Tree SHAP para la clase conceptual LoanApproved=1."""
    estimator = pipeline.named_steps["estimador"]
    positive_index = localizar_clase_positiva(estimator)
    explainer = shap.TreeExplainer(estimator)
    explanation = explainer(matrix)
    values = _seleccionar_salida_positiva(
        explanation.values,
        positive_index=positive_index,
        n_classes=len(estimator.classes_),
        n_samples=matrix.shape[0],
        n_features=matrix.shape[1],
    )
    base_values = _seleccionar_base_value(
        explanation.base_values,
        positive_index=positive_index,
        n_classes=len(estimator.classes_),
        n_samples=matrix.shape[0],
    )
    return values, base_values


def _factor_local(
    descriptor: FeatureDescriptor,
    engineered_row: pd.Series,
    encoded_value: float,
    shap_value: float,
) -> dict[str, Any]:
    application_value = _json_value(engineered_row[descriptor.model_feature])
    category_active = None if descriptor.category is None else bool(encoded_value == 1.0)
    feature_label = descriptor.display_name
    if category_active is not None:
        state = "present" if category_active else "absent"
        feature_label = f"{feature_label} ({state})"
    return {
        "feature": feature_label,
        "model_feature": descriptor.model_feature,
        "source_features": list(descriptor.source_features),
        "category": descriptor.category,
        "raw_code": descriptor.category,
        "category_label": descriptor.category_label,
        "category_active": category_active,
        "application_value": application_value,
        "encoded_value": float(encoded_value),
        "shap_value": float(shap_value),
    }


def explicar_solicitud(
    pipeline: Any,
    raw_application: pd.DataFrame | pd.Series | dict[str, Any],
    *,
    top_n: int = 5,
    summary: dict[str, Any] | None = None,
    global_feature_context: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Devuelve evidencia local estructurada e independiente de cualquier LLM."""
    if top_n < 1:
        raise ValueError("top_n debe ser positivo")
    if isinstance(raw_application, dict):
        raw = pd.DataFrame([raw_application], columns=MODEL_PREDICTOR_COLUMNS)
    elif isinstance(raw_application, pd.Series):
        raw = raw_application.to_frame().T.loc[:, MODEL_PREDICTOR_COLUMNS]
    else:
        raw = raw_application.copy()
    if len(raw) != 1:
        raise ValueError("explicar_solicitud requiere exactamente una fila")

    engineered, matrix, descriptors = _transformar_para_estimador(pipeline, raw)
    shap_values, base_values = calcular_shap(pipeline, matrix)
    estimator = pipeline.named_steps["estimador"]
    positive_index = localizar_clase_positiva(estimator)
    prediction = int(pipeline.predict(raw)[0])
    probability = float(pipeline.predict_proba(raw)[0, positive_index])

    factors = [
        _factor_local(
            descriptor,
            engineered.iloc[0],
            matrix[0, descriptor.index],
            shap_values[0, descriptor.index],
        )
        for descriptor in descriptors
    ]
    positive = sorted(
        (factor for factor in factors if factor["shap_value"] > 0),
        key=lambda factor: factor["shap_value"],
        reverse=True,
    )[:top_n]
    negative = sorted(
        (factor for factor in factors if factor["shap_value"] < 0),
        key=lambda factor: factor["shap_value"],
    )[:top_n]

    summary = summary or {}
    champion = summary.get("champion_final", {})
    registry = champion.get("registry", {})
    return {
        "prediction": prediction,
        "prediction_label": "approved" if prediction == POSITIVE_CLASS else "denied",
        "positive_class": POSITIVE_CLASS,
        "probability": probability,
        "base_value": float(base_values[0]),
        "shap_output_space": "CatBoost raw margin (log-odds)",
        "top_positive_factors": positive,
        "top_negative_factors": negative,
        "global_feature_context": global_feature_context,
        "model": {
            "name": registry.get("name", REGISTERED_MODEL_NAME),
            "version": registry.get("version"),
            "selected_configuration": champion.get("selected_from"),
        },
        "dataset": {
            "version": summary.get("dataset_version", DATASET_VERSION),
            "sha256": summary.get("dataset_sha256"),
        },
        "warning": NON_CAUSALITY_WARNING,
    }


def _global_importance_tables(
    values: np.ndarray, descriptors: list[FeatureDescriptor]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    transformed = pd.DataFrame(
        [
            {
                "transformed_feature": item.transformed_feature,
                "display_feature": item.display_name,
                "model_feature": item.model_feature,
                "business_feature": item.business_feature,
                "source_features": "|".join(item.source_features),
                "category": item.category,
                "raw_code": item.category,
                "category_label": item.category_label,
                "mean_abs_shap": float(np.abs(values[:, item.index]).mean()),
            }
            for item in descriptors
        ]
    ).sort_values("mean_abs_shap", ascending=False, ignore_index=True)
    transformed.insert(0, "rank", np.arange(1, len(transformed) + 1))

    grouped_rows = []
    for model_feature in dict.fromkeys(item.model_feature for item in descriptors):
        feature_descriptors = [
            item for item in descriptors if item.model_feature == model_feature
        ]
        indices = [item.index for item in feature_descriptors]
        grouped_shap_per_row = values[:, indices].sum(axis=1)
        first = feature_descriptors[0]
        grouped_rows.append(
            {
                "model_feature": model_feature,
                "business_feature": first.business_feature,
                "source_features": "|".join(first.source_features),
                "transformed_feature_count": len(indices),
                "mean_abs_shap": float(np.abs(grouped_shap_per_row).mean()),
            }
        )
    grouped = pd.DataFrame(grouped_rows).sort_values(
        "mean_abs_shap", ascending=False, ignore_index=True
    )
    grouped.insert(0, "rank", np.arange(1, len(grouped) + 1))
    return grouped, transformed


def _grouped_shap_matrix(
    values: np.ndarray, descriptors: list[FeatureDescriptor]
) -> tuple[np.ndarray, list[str]]:
    model_features = list(dict.fromkeys(item.model_feature for item in descriptors))
    grouped_values = np.column_stack(
        [
            values[
                :,
                [
                    item.index
                    for item in descriptors
                    if item.model_feature == model_feature
                ],
            ].sum(axis=1)
            for model_feature in model_features
        ]
    )
    business_names = [
        next(
            item.business_feature
            for item in descriptors
            if item.model_feature == model_feature
        )
        for model_feature in model_features
    ]
    return grouped_values, business_names


def generar_artifacts_xai(
    pipeline: Any,
    data: pd.DataFrame,
    output_dir: Path,
    *,
    sample_size: int = DEFAULT_GLOBAL_SAMPLE_SIZE,
    top_n: int = DEFAULT_TOP_N,
    summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Genera artifacts globales y un ejemplo local sobre una muestra de training."""
    if sample_size < 1 or top_n < 1:
        raise ValueError("sample_size y top_n deben ser positivos")
    X_train, _, _, _ = separar_train_test_v2(data)
    actual_size = min(sample_size, len(X_train))
    sample = X_train.sample(n=actual_size, random_state=RANDOM_STATE)
    _, matrix, descriptors = _transformar_para_estimador(pipeline, sample)
    values, _ = calcular_shap(pipeline, matrix)
    grouped, transformed = _global_importance_tables(values, descriptors)

    output_dir.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(output_dir / "global_feature_importance.csv", index=False)
    transformed.to_csv(
        output_dir / "global_feature_importance_transformed.csv", index=False
    )
    grouped_records = grouped.to_dict(orient="records")
    transformed_records = transformed.to_dict(orient="records")
    _write_json(output_dir / "global_feature_importance.json", grouped_records)
    _write_json(
        output_dir / "global_feature_importance_transformed.json",
        transformed_records,
    )

    grouped_values, grouped_names = _grouped_shap_matrix(values, descriptors)
    shap.summary_plot(
        grouped_values,
        feature_names=grouped_names,
        max_display=top_n,
        show=False,
    )
    plt.gcf().tight_layout()
    plt.gcf().savefig(output_dir / "shap_summary.png", dpi=160, bbox_inches="tight")
    plt.close()
    top_grouped = grouped.head(top_n).sort_values("mean_abs_shap")
    figure, axis = plt.subplots(figsize=(9, 6))
    axis.barh(top_grouped["business_feature"], top_grouped["mean_abs_shap"])
    axis.set_xlabel("mean(abs(sum(SHAP per row by business feature)))")
    axis.set_title("Global grouped SHAP importance")
    figure.tight_layout()
    figure.savefig(output_dir / "shap_bar.png", dpi=160, bbox_inches="tight")
    plt.close(figure)

    context = [
        {
            key: _json_value(value)
            for key, value in record.items()
        }
        for record in grouped_records[:top_n]
    ]
    evidence = explicar_solicitud(
        pipeline,
        sample.iloc[[0]],
        top_n=min(5, top_n),
        summary=summary,
        global_feature_context=context,
    )
    _write_json(output_dir / "local_example.json", evidence)

    metadata = {
        "method": "Tree SHAP",
        "shap_version": version("shap"),
        "model_name": evidence["model"]["name"],
        "model_version": evidence["model"]["version"],
        "dataset_version": evidence["dataset"]["version"],
        "dataset_sha256": evidence["dataset"]["sha256"],
        "sample_source": "training partition after deterministic 80/20 split",
        "sample_size": actual_size,
        "random_state": RANDOM_STATE,
        "top_n": top_n,
        "positive_class": POSITIVE_CLASS,
        "transformed_feature_count": len(descriptors),
        "grouped_feature_count": len(grouped),
        "global_aggregation": (
            "sum transformed SHAP contributions by model feature within each "
            "row, then mean absolute value across rows"
        ),
        "hmda_category_label_source": (
            "CFPB 2023 Reportable HMDA Data regulatory and reporting overview"
        ),
        "hmda_category_mappings": HMDA_CATEGORY_LABELS,
        "audit_only_excluded": list(AUDIT_ONLY_COLUMNS),
        "warning": NON_CAUSALITY_WARNING,
    }
    _write_json(output_dir / "metadata.json", metadata)
    return {"metadata": metadata, "top_features": context, "evidence": evidence}


def registrar_artifacts_xai(
    output_dir: Path,
    tracking: TrackingConfig,
    metadata: dict[str, Any],
) -> dict[str, str]:
    validar_run_name(XAI_RUN_NAME)
    with mlflow.start_run(
        experiment_id=tracking.experiment_id,
        run_name=XAI_RUN_NAME,
        tags={
            "model_family": "catboost",
            "stage": "xai",
            "dataset_version": metadata["dataset_version"],
            "selection_status": "champion_explanation",
            "xai_method": "shap",
        },
    ) as run:
        mlflow.log_params(
            {
                "model_name": metadata["model_name"],
                "model_version": metadata["model_version"],
                "xai_method": metadata["method"],
                "global_sample_size": metadata["sample_size"],
                "top_n": metadata["top_n"],
                "random_state": metadata["random_state"],
                "positive_class": metadata["positive_class"],
                "dataset_sha256": metadata["dataset_sha256"],
                "global_aggregation": metadata["global_aggregation"],
            }
        )
        if tracking.backend == "dagshub" and tracking.ui_url:
            run_url = (
                f"{tracking.ui_url}/#/experiments/{tracking.experiment_id}"
                f"/runs/{run.info.run_id}"
            )
        else:
            run_url = tracking.ui_url or ""
        run_metadata = {
            "run_id": run.info.run_id,
            "run_name": XAI_RUN_NAME,
            "backend": tracking.backend,
            "url": run_url,
        }
        _write_json(output_dir / "run_metadata.json", run_metadata)
        mlflow.log_artifacts(str(output_dir), artifact_path="xai")
        return run_metadata


def ejecutar_xai(
    *,
    dataset_path: Path = DEFAULT_DATASET_PATH,
    champion_path: Path = DEFAULT_CHAMPION_PATH,
    summary_path: Path = DEFAULT_SUMMARY_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    sample_size: int = DEFAULT_GLOBAL_SAMPLE_SIZE,
    top_n: int = DEFAULT_TOP_N,
    backend: str | None = None,
    log_mlflow: bool = True,
) -> dict[str, Any]:
    pipeline = cargar_champion(champion_path)
    summary = cargar_resumen(summary_path)
    data = pd.read_csv(dataset_path)
    result = generar_artifacts_xai(
        pipeline,
        data,
        output_dir,
        sample_size=sample_size,
        top_n=top_n,
        summary=summary,
    )
    if log_mlflow:
        load_dotenv(ROOT / ".env", override=False)
        tracking = configurar_mlflow(
            ROOT,
            backend=backend,
            experiment_name=EXPERIMENT_NAME,
        )
        result["tracking"] = registrar_artifacts_xai(
            output_dir, tracking, result["metadata"]
        )
    return result


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-size", type=int, default=DEFAULT_GLOBAL_SAMPLE_SIZE)
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--backend", choices=("local", "dagshub"), default=None)
    parser.add_argument("--skip-mlflow", action="store_true")
    args = parser.parse_args()
    result = ejecutar_xai(
        sample_size=args.sample_size,
        top_n=args.top_n,
        backend=args.backend,
        log_mlflow=not args.skip_mlflow,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
