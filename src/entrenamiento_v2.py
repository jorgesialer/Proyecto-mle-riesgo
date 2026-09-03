"""Benchmark reproducible V2 con MLflow y holdout final reservado."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    average_precision_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from xgboost import XGBClassifier

from src.mlflow_utils import TrackingConfig, configurar_mlflow, validar_run_name
from src.preprocesamiento_v2 import (
    FEATURE_CONFIG_FULL,
    FEATURE_CONFIG_WITHOUT_WORKFLOW,
    RANDOM_STATE,
    WORKFLOW_RAW_COLUMNS,
    crear_pipeline_v2,
    obtener_columnas_modelo_v2,
    separar_train_test_v2,
)


DATASET_VERSION = "hmda_2023_loan_approval_v2"
EXPERIMENT_NAME = "credit-approval-v2-benchmark"
REGISTERED_MODEL_NAME = "credit-approval-v2"
CV_FOLDS = 5
POSITIVE_CLASS = 1
TRACKED_FEATURES = (
    "total_units",
    "loan_to_property_value",
    "non_amortizing_feature_count",
)

SCORING = {
    "precision": make_scorer(precision_score, pos_label=POSITIVE_CLASS, zero_division=0),
    "recall": make_scorer(recall_score, pos_label=POSITIVE_CLASS, zero_division=0),
    "f1": make_scorer(f1_score, pos_label=POSITIVE_CLASS, zero_division=0),
    "roc_auc": "roc_auc",
    "average_precision": "average_precision",
}


@dataclass(frozen=True)
class RunSpec:
    run_name: str
    model_family: str
    feature_config: str
    stage: str
    params: dict[str, Any]


def construir_run_specs() -> tuple[RunSpec, ...]:
    """Historia experimental acotada: 6 comparaciones + 5 tunings."""
    return (
        RunSpec(
            "rf_baseline_full",
            "random_forest",
            FEATURE_CONFIG_FULL,
            "baseline",
            {"n_estimators": 250, "max_depth": None, "min_samples_leaf": 1},
        ),
        RunSpec(
            "rf_robust_without_workflow_features",
            "random_forest",
            FEATURE_CONFIG_WITHOUT_WORKFLOW,
            "robustness",
            {"n_estimators": 250, "max_depth": None, "min_samples_leaf": 1},
        ),
        RunSpec(
            "rf_tuned_01",
            "random_forest",
            FEATURE_CONFIG_FULL,
            "tuning",
            {
                "n_estimators": 400,
                "max_depth": 18,
                "min_samples_leaf": 2,
                "max_features": 0.7,
            },
        ),
        RunSpec(
            "xgb_baseline_full",
            "xgboost",
            FEATURE_CONFIG_FULL,
            "baseline",
            {
                "n_estimators": 250,
                "max_depth": 5,
                "learning_rate": 0.05,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
            },
        ),
        RunSpec(
            "xgb_robust_without_workflow_features",
            "xgboost",
            FEATURE_CONFIG_WITHOUT_WORKFLOW,
            "robustness",
            {
                "n_estimators": 250,
                "max_depth": 5,
                "learning_rate": 0.05,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
            },
        ),
        RunSpec(
            "xgb_tuned_01",
            "xgboost",
            FEATURE_CONFIG_FULL,
            "tuning",
            {
                "n_estimators": 350,
                "max_depth": 3,
                "learning_rate": 0.07,
                "subsample": 0.8,
                "colsample_bytree": 0.85,
            },
        ),
        RunSpec(
            "xgb_tuned_02",
            "xgboost",
            FEATURE_CONFIG_FULL,
            "tuning",
            {
                "n_estimators": 450,
                "max_depth": 4,
                "learning_rate": 0.04,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
            },
        ),
        RunSpec(
            "catboost_baseline_full",
            "catboost",
            FEATURE_CONFIG_FULL,
            "baseline",
            {"iterations": 250, "depth": 6, "learning_rate": 0.05},
        ),
        RunSpec(
            "catboost_robust_without_workflow_features",
            "catboost",
            FEATURE_CONFIG_WITHOUT_WORKFLOW,
            "robustness",
            {"iterations": 250, "depth": 6, "learning_rate": 0.05},
        ),
        RunSpec(
            "catboost_tuned_01",
            "catboost",
            FEATURE_CONFIG_FULL,
            "tuning",
            {"iterations": 350, "depth": 4, "learning_rate": 0.06},
        ),
        RunSpec(
            "catboost_tuned_02",
            "catboost",
            FEATURE_CONFIG_FULL,
            "tuning",
            {"iterations": 450, "depth": 7, "learning_rate": 0.04},
        ),
    )


def crear_estimador(spec: RunSpec):
    """Construye exclusivamente una de las tres familias aprobadas."""
    if spec.model_family == "random_forest":
        return RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_jobs=-1,
            **spec.params,
        )
    if spec.model_family == "xgboost":
        return XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            tree_method="hist",
            **spec.params,
        )
    if spec.model_family == "catboost":
        return CatBoostClassifier(
            loss_function="Logloss",
            random_seed=RANDOM_STATE,
            verbose=False,
            allow_writing_files=False,
            thread_count=-1,
            **spec.params,
        )
    raise ValueError(f"Familia no aprobada: {spec.model_family!r}")


def _summary_cv(cv_result: dict[str, Any]) -> tuple[dict[str, float], dict[str, Any]]:
    metrics: dict[str, float] = {}
    folds: dict[str, Any] = {}
    for metric_name in SCORING:
        values = np.asarray(cv_result[f"test_{metric_name}"], dtype=float)
        metrics[f"cv_{metric_name}_mean"] = float(values.mean())
        metrics[f"cv_{metric_name}_std"] = float(values.std(ddof=0))
        folds[metric_name] = values.tolist()
    for time_name in ("fit_time", "score_time"):
        values = np.asarray(cv_result[time_name], dtype=float)
        metrics[f"{time_name}_mean"] = float(values.mean())
        metrics[f"{time_name}_std"] = float(values.std(ddof=0))
        folds[time_name] = values.tolist()
    return metrics, folds


def _base_feature_name(encoded_name: str, candidates: tuple[str, ...]) -> str:
    if encoded_name in candidates:
        return encoded_name
    for candidate in sorted(candidates, key=len, reverse=True):
        if encoded_name.startswith(candidate + "_"):
            return candidate
    return encoded_name


def _aggregate_importances(estimators: list[Any]) -> tuple[pd.DataFrame, dict[str, float]]:
    fold_importances: list[dict[str, float]] = []
    for pipeline in estimators:
        preprocessing = pipeline.named_steps["preprocesamiento"]
        model = pipeline.named_steps["estimador"]
        values = np.asarray(model.feature_importances_, dtype=float)
        total_importance = values.sum()
        if total_importance > 0:
            values = values / total_importance
        names = np.asarray(preprocessing.get_feature_names_out(), dtype=object)
        if len(values) != len(names):
            raise ValueError("Importancias y nombres transformados no coinciden")
        numeric_columns, categorical_columns = obtener_columnas_modelo_v2(
            _feature_config_from_pipeline(pipeline)
        )
        candidates = numeric_columns + categorical_columns
        grouped: dict[str, float] = {}
        for name, value in zip(names, values):
            base = _base_feature_name(str(name), candidates)
            grouped[base] = grouped.get(base, 0.0) + float(value)
        fold_importances.append(grouped)

    all_features = sorted(set().union(*(fold.keys() for fold in fold_importances)))
    rows = []
    for feature in all_features:
        values = [fold.get(feature, 0.0) for fold in fold_importances]
        rows.append(
            {
                "feature": feature,
                "importance_mean": float(np.mean(values)),
                "importance_std": float(np.std(values, ddof=0)),
            }
        )
    frame = pd.DataFrame(rows).sort_values("importance_mean", ascending=False)
    tracked = {
        feature: float(
            frame.loc[frame["feature"] == feature, "importance_mean"].sum()
        )
        for feature in TRACKED_FEATURES
    }
    return frame, tracked


def _feature_config_from_pipeline(pipeline: Any) -> str:
    numeric = tuple(
        pipeline.named_steps["preprocesamiento"].transformers[0][2]
    )
    full_numeric, _ = obtener_columnas_modelo_v2(FEATURE_CONFIG_FULL)
    return FEATURE_CONFIG_FULL if numeric == full_numeric else FEATURE_CONFIG_WITHOUT_WORKFLOW


def _log_importance_plot(frame: pd.DataFrame) -> None:
    top = frame.head(15).sort_values("importance_mean")
    figure, axis = plt.subplots(figsize=(9, 6))
    axis.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"])
    axis.set_title("Mean feature importance across CV folds")
    axis.set_xlabel("Importance")
    figure.tight_layout()
    mlflow.log_figure(figure, "plots/feature_importance_cv.png")
    plt.close(figure)


def ejecutar_run_cv(
    spec: RunSpec,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_folds: int,
    tracking: TrackingConfig,
    dataset_metadata_path: Path,
    dataset_sha256: str,
) -> dict[str, Any]:
    validar_run_name(spec.run_name)
    estimator = crear_estimador(spec)
    pipeline = crear_pipeline_v2(estimator, feature_config=spec.feature_config)
    splitter = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    numeric_columns, categorical_columns = obtener_columnas_modelo_v2(
        spec.feature_config
    )

    with mlflow.start_run(
        experiment_id=tracking.experiment_id,
        run_name=spec.run_name,
        tags={
            "model_family": spec.model_family,
            "feature_config": spec.feature_config,
            "stage": spec.stage,
            "dataset_version": DATASET_VERSION,
            "selection_status": "candidate",
        },
    ) as run:
        mlflow.log_params(
            {
                **spec.params,
                "model_family": spec.model_family,
                "feature_config": spec.feature_config,
                "cv_folds": cv_folds,
                "random_seed": RANDOM_STATE,
                "positive_class": POSITIVE_CLASS,
                "training_rows": len(X_train),
                "dataset_sha256": dataset_sha256,
            }
        )
        mlflow.log_dict(
            {
                "numeric": list(numeric_columns),
                "categorical": list(categorical_columns),
                "excluded_raw": (
                    list(WORKFLOW_RAW_COLUMNS)
                    if spec.feature_config == FEATURE_CONFIG_WITHOUT_WORKFLOW
                    else []
                ),
            },
            "feature_configuration.json",
        )
        mlflow.log_dict(
            {
                "counts": {str(key): int(value) for key, value in y_train.value_counts().items()},
                "proportions": {
                    str(key): float(value) for key, value in y_train.value_counts(normalize=True).items()
                },
            },
            "training_target_distribution.json",
        )
        mlflow.log_artifact(str(dataset_metadata_path), artifact_path="dataset")

        cv_result = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=splitter,
            scoring=SCORING,
            n_jobs=1,
            return_train_score=False,
            return_estimator=True,
            error_score="raise",
        )
        metrics, fold_metrics = _summary_cv(cv_result)
        mlflow.log_metrics(metrics)
        mlflow.log_dict(fold_metrics, "cv_metrics_by_fold.json")

        importance_frame, tracked_importances = _aggregate_importances(
            list(cv_result["estimator"])
        )
        mlflow.log_table(importance_frame, "feature_importance_cv.json")
        _log_importance_plot(importance_frame)
        mlflow.log_metrics(
            {
                f"importance_{feature}": value
                for feature, value in tracked_importances.items()
            }
        )

        return {
            "run_id": run.info.run_id,
            "run_name": spec.run_name,
            "model_family": spec.model_family,
            "feature_config": spec.feature_config,
            "stage": spec.stage,
            "params": spec.params,
            "metrics": metrics,
            "tracked_feature_importance": tracked_importances,
        }


def seleccionar_champion(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Selecciona solo por CV: F1, luego AP y ROC-AUC."""
    if not results:
        raise ValueError("No hay resultados CV para seleccionar champion")
    return max(
        results,
        key=lambda result: (
            result["metrics"]["cv_f1_mean"],
            result["metrics"]["cv_average_precision_mean"],
            result["metrics"]["cv_roc_auc_mean"],
        ),
    )


def _holdout_metrics(y_true: pd.Series, predictions: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    return {
        "holdout_precision": float(precision_score(y_true, predictions, zero_division=0)),
        "holdout_recall": float(recall_score(y_true, predictions, zero_division=0)),
        "holdout_f1": float(f1_score(y_true, predictions, zero_division=0)),
        "holdout_roc_auc": float(roc_auc_score(y_true, probabilities)),
        "holdout_average_precision": float(average_precision_score(y_true, probabilities)),
    }


def serializar_pipeline(pipeline: Any, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as file:
        pickle.dump(pipeline, file)


def _log_holdout_plots(y_true: pd.Series, predictions: np.ndarray, probabilities: np.ndarray) -> list[list[int]]:
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    figure, axis = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay(matrix, display_labels=[0, 1]).plot(ax=axis, colorbar=False)
    axis.set_title("Champion final holdout")
    figure.tight_layout()
    mlflow.log_figure(figure, "plots/confusion_matrix_holdout.png")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_true, probabilities, ax=axis)
    axis.set_title("ROC curve - final holdout")
    figure.tight_layout()
    mlflow.log_figure(figure, "plots/roc_curve_holdout.png")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6, 5))
    PrecisionRecallDisplay.from_predictions(y_true, probabilities, ax=axis)
    axis.set_title("Precision-Recall curve - final holdout")
    figure.tight_layout()
    mlflow.log_figure(figure, "plots/precision_recall_curve_holdout.png")
    plt.close(figure)
    return matrix.astype(int).tolist()


def evaluar_champion_holdout(
    selected: dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    tracking: TrackingConfig,
    dataset_metadata_path: Path,
    dataset_sha256: str,
    champion_output_path: Path,
) -> dict[str, Any]:
    spec = RunSpec(
        run_name=selected["run_name"],
        model_family=selected["model_family"],
        feature_config=selected["feature_config"],
        stage=selected["stage"],
        params=selected["params"],
    )
    final_run_name = f"champion_{spec.model_family}_final"
    validar_run_name(final_run_name)
    pipeline = crear_pipeline_v2(
        crear_estimador(spec),
        feature_config=spec.feature_config,
    )
    numeric_columns, categorical_columns = obtener_columnas_modelo_v2(
        spec.feature_config
    )

    with mlflow.start_run(
        experiment_id=tracking.experiment_id,
        run_name=final_run_name,
        tags={
            "model_family": spec.model_family,
            "feature_config": spec.feature_config,
            "stage": "final_holdout",
            "dataset_version": DATASET_VERSION,
            "selection_status": "champion",
            "selected_from_run_id": selected["run_id"],
        },
    ) as run:
        mlflow.log_params(
            {
                **spec.params,
                "model_family": spec.model_family,
                "feature_config": spec.feature_config,
                "random_seed": RANDOM_STATE,
                "positive_class": POSITIVE_CLASS,
                "training_rows": len(X_train),
                "holdout_rows": len(X_test),
                "dataset_sha256": dataset_sha256,
            }
        )
        fit_started = time.perf_counter()
        pipeline.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - fit_started
        predictions = pipeline.predict(X_test)
        probabilities = pipeline.predict_proba(X_test)[:, 1]
        metrics = _holdout_metrics(y_test, predictions, probabilities)
        metrics["final_fit_time"] = fit_seconds
        mlflow.log_metrics(metrics)

        matrix = _log_holdout_plots(y_test, predictions, probabilities)
        mlflow.log_dict(
            {"labels": [0, 1], "matrix": matrix},
            "confusion_matrix_holdout.json",
        )
        mlflow.log_dict(
            {
                "numeric": list(numeric_columns),
                "categorical": list(categorical_columns),
                "raw_predictor_columns": list(X_train.columns),
                "excluded_raw": (
                    list(WORKFLOW_RAW_COLUMNS)
                    if spec.feature_config == FEATURE_CONFIG_WITHOUT_WORKFLOW
                    else []
                ),
            },
            "feature_configuration.json",
        )
        mlflow.log_artifact(str(dataset_metadata_path), artifact_path="dataset")
        serializar_pipeline(pipeline, champion_output_path)
        mlflow.log_artifact(str(champion_output_path), artifact_path="serialized_pipeline")

        input_example = X_train.head(5).copy()
        integer_columns = input_example.select_dtypes(include=["integer"]).columns
        for column in integer_columns:
            input_example[column] = input_example[column].astype(float)
        signature = infer_signature(input_example, pipeline.predict(input_example))
        pip_requirements = [
            f"mlflow=={version('mlflow')}",
            f"scikit-learn=={version('scikit-learn')}",
            f"pandas=={version('pandas')}",
            f"numpy=={version('numpy')}",
            f"xgboost=={version('xgboost')}",
            f"catboost=={version('catboost')}",
            f"cloudpickle=={version('cloudpickle')}",
        ]
        model_info = mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model",
            serialization_format="cloudpickle",
            signature=signature,
            input_example=input_example,
            code_paths=[str(Path.cwd() / "src")],
            pip_requirements=pip_requirements,
            metadata={
                "dataset_version": DATASET_VERSION,
                "dataset_sha256": dataset_sha256,
                "positive_class": POSITIVE_CLASS,
            },
        )

        registry = {"used": False, "name": None, "version": None, "error": None}
        if tracking.registry_supported:
            try:
                registered = mlflow.register_model(
                    model_uri=model_info.model_uri,
                    name=REGISTERED_MODEL_NAME,
                )
                registry.update(
                    {
                        "used": True,
                        "name": REGISTERED_MODEL_NAME,
                        "version": str(registered.version),
                    }
                )
                mlflow.set_tag("registry_status", "registered")
            except Exception as exc:  # backend capability is external
                registry["error"] = f"{type(exc).__name__}: {exc}"
                mlflow.set_tag("registry_status", "unsupported_or_failed")

        return {
            "run_id": run.info.run_id,
            "run_name": final_run_name,
            "selected_from": selected["run_name"],
            "model_family": spec.model_family,
            "feature_config": spec.feature_config,
            "params": spec.params,
            "metrics": metrics,
            "confusion_matrix": matrix,
            "pipeline_path": str(champion_output_path),
            "mlflow_model_uri": model_info.model_uri,
            "registry": registry,
        }


def _robustness_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for family in ("random_forest", "xgboost", "catboost"):
        full = next(
            result
            for result in results
            if result["model_family"] == family and result["stage"] == "baseline"
        )
        robust = next(
            result
            for result in results
            if result["model_family"] == family and result["stage"] == "robustness"
        )
        summary[family] = {
            "full_run": full["run_name"],
            "robust_run": robust["run_name"],
            "delta_cv_f1_robust_minus_full": (
                robust["metrics"]["cv_f1_mean"] - full["metrics"]["cv_f1_mean"]
            ),
            "delta_cv_average_precision_robust_minus_full": (
                robust["metrics"]["cv_average_precision_mean"]
                - full["metrics"]["cv_average_precision_mean"]
            ),
            "delta_cv_roc_auc_robust_minus_full": (
                robust["metrics"]["cv_roc_auc_mean"]
                - full["metrics"]["cv_roc_auc_mean"]
            ),
        }
    return summary


def _feature_signal_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    baselines = {
        result["model_family"]: result
        for result in results
        if result["stage"] == "baseline" and result["feature_config"] == FEATURE_CONFIG_FULL
    }
    importances = {
        feature: {
            family: baseline["tracked_feature_importance"][feature]
            for family, baseline in baselines.items()
        }
        for feature in TRACKED_FEATURES
    }
    return {
        "normalized_importance_by_family": importances,
        "assessment": {
            "loan_to_property_value": (
                "consistent signal in all three families; retain"
            ),
            "total_units": (
                "consistent but low signal; retain for a future controlled ablation"
            ),
            "non_amortizing_feature_count": (
                "consistent but low signal; future ablation should test whether it justifies complexity"
            ),
        },
    }


def ejecutar_benchmark(
    dataset_path: Path,
    dataset_metadata_path: Path,
    summary_path: Path,
    champion_output_path: Path,
    cv_folds: int = CV_FOLDS,
) -> dict[str, Any]:
    project_root = Path.cwd()
    metadata = json.loads(dataset_metadata_path.read_text(encoding="utf-8"))
    dataset_sha256 = metadata["output"]["sha256"]
    data = pd.read_csv(dataset_path)
    X_train, X_test, y_train, y_test = separar_train_test_v2(data)
    tracking = configurar_mlflow(project_root, experiment_name=EXPERIMENT_NAME)

    results = []
    for spec in construir_run_specs():
        print(f"[CV] {spec.run_name}", flush=True)
        result = ejecutar_run_cv(
            spec,
            X_train,
            y_train,
            cv_folds,
            tracking,
            dataset_metadata_path,
            dataset_sha256,
        )
        results.append(result)
        print(
            f"     F1={result['metrics']['cv_f1_mean']:.6f} "
            f"AP={result['metrics']['cv_average_precision_mean']:.6f}",
            flush=True,
        )

    selected = seleccionar_champion(results)
    print(f"[SELECTED BY CV] {selected['run_name']}", flush=True)
    final_result = evaluar_champion_holdout(
        selected,
        X_train,
        y_train,
        X_test,
        y_test,
        tracking,
        dataset_metadata_path,
        dataset_sha256,
        champion_output_path,
    )

    summary = {
        "dataset_version": DATASET_VERSION,
        "dataset_sha256": dataset_sha256,
        "positive_class": POSITIVE_CLASS,
        "selection_criterion": [
            "cv_f1_mean",
            "cv_average_precision_mean",
            "cv_roc_auc_mean",
        ],
        "protocol": {
            "train_rows": len(X_train),
            "holdout_rows": len(X_test),
            "cv_folds": cv_folds,
            "random_state": RANDOM_STATE,
            "holdout_used_for_selection": False,
        },
        "tracking": asdict(tracking),
        "cv_runs": results,
        "robustness": _robustness_summary(results),
        "feature_signal": _feature_signal_summary(results),
        "selected_cv_run": selected,
        "champion_final": final_result,
        "total_runs": len(results) + 1,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary["champion_final"], indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/hmda_2023_loan_approval_v2.csv"),
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("data/hmda_2023_loan_approval_v2.metadata.json"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("artifacts/benchmark_v2_summary.json"),
    )
    parser.add_argument(
        "--champion-output",
        type=Path,
        default=Path("artifacts/pipeline_champion_v2.pkl"),
    )
    parser.add_argument("--cv-folds", type=int, default=CV_FOLDS)
    args = parser.parse_args()
    ejecutar_benchmark(
        args.dataset,
        args.metadata,
        args.summary,
        args.champion_output,
        cv_folds=args.cv_folds,
    )


if __name__ == "__main__":
    main()
