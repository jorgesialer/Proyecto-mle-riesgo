"""Configuracion reutilizable de MLflow local o DagsHub sin secretos embebidos."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

import mlflow


DEFAULT_EXPERIMENT_NAME = "credit-approval-v2-benchmark"
ALLOWED_BACKENDS = frozenset({"local", "dagshub"})
REQUIRED_DAGSHUB_ENV = (
    "MLFLOW_TRACKING_URI",
    "MLFLOW_TRACKING_USERNAME",
    "MLFLOW_TRACKING_PASSWORD",
)


@dataclass(frozen=True)
class TrackingConfig:
    backend: str
    tracking_uri: str
    experiment_name: str
    experiment_id: str
    ui_url: str | None
    registry_supported: bool


def validar_run_name(run_name: str) -> None:
    """Evita nombres opacos o generados aleatoriamente como identificador principal."""
    if not re.fullmatch(r"[a-z0-9]+(?:_[a-z0-9]+){2,}", run_name):
        raise ValueError(
            "run_name debe ser descriptivo, en snake_case y contener al menos "
            "tres segmentos"
        )


def _get_or_create_experiment(
    experiment_name: str,
    artifact_location: str | None = None,
) -> str:
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is not None:
        return experiment.experiment_id
    return mlflow.create_experiment(
        experiment_name,
        artifact_location=artifact_location,
    )


def configurar_mlflow(
    project_root: Path,
    backend: str | None = None,
    experiment_name: str | None = None,
) -> TrackingConfig:
    """Configura tracking local SQLite o un servidor DagsHub ya autenticado."""
    selected_backend = (backend or os.getenv("MLFLOW_BACKEND", "local")).lower()
    if selected_backend not in ALLOWED_BACKENDS:
        raise ValueError(
            f"MLFLOW_BACKEND debe ser uno de {sorted(ALLOWED_BACKENDS)}; "
            f"se recibio {selected_backend!r}"
        )
    selected_experiment = experiment_name or os.getenv(
        "MLFLOW_EXPERIMENT_NAME", DEFAULT_EXPERIMENT_NAME
    )

    if selected_backend == "dagshub":
        missing = [name for name in REQUIRED_DAGSHUB_ENV if not os.getenv(name)]
        if missing:
            raise EnvironmentError(
                "Backend DagsHub solicitado pero faltan variables: "
                + ", ".join(missing)
            )
        tracking_uri = os.environ["MLFLOW_TRACKING_URI"]
        mlflow.set_tracking_uri(tracking_uri)
        experiment_id = _get_or_create_experiment(selected_experiment)
        return TrackingConfig(
            backend="dagshub",
            tracking_uri=tracking_uri,
            experiment_name=selected_experiment,
            experiment_id=experiment_id,
            ui_url=tracking_uri,
            registry_supported=True,
        )

    database_path = Path(
        os.getenv("MLFLOW_LOCAL_DB", str(project_root / "mlflow.db"))
    ).resolve()
    artifact_root = Path(
        os.getenv("MLFLOW_LOCAL_ARTIFACT_ROOT", str(project_root / "mlartifacts"))
    ).resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)
    tracking_uri = f"sqlite:///{database_path.as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    experiment_id = _get_or_create_experiment(
        selected_experiment,
        artifact_location=artifact_root.as_uri(),
    )
    return TrackingConfig(
        backend="local",
        tracking_uri=tracking_uri,
        experiment_name=selected_experiment,
        experiment_id=experiment_id,
        ui_url="http://127.0.0.1:5000",
        registry_supported=True,
    )
