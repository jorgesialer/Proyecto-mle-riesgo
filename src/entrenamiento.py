"""Entrenamiento reproducible y fold-safe del baseline Random Forest."""

from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

from .preprocesamiento import (
    COLUMNAS_PREDICTORAS,
    TARGET,
    crear_pipeline_baseline,
    validar_esquema_entrenamiento,
)


RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_SPLITS = 5
SCORING = {
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
}


def validar_target(target: pd.Series) -> None:
    """Exige un target binario {0, 1} sin valores faltantes."""
    if target.isna().any():
        raise ValueError(f"{TARGET} no puede contener valores nulos")
    valores = set(target.unique().tolist())
    if valores != {0, 1}:
        raise ValueError(
            f"{TARGET} debe contener exclusivamente las clases 0 y 1; "
            f"valores encontrados: {sorted(valores)}"
        )


class EntrenadorModelo:
    def __init__(self, ruta_datos: str, ruta_pipeline: str):
        self.ruta_datos = Path(ruta_datos)
        self.ruta_pipeline = Path(ruta_pipeline)

    def ejecutar(self):
        print("1. Cargando datos crudos...")
        df = pd.read_csv(self.ruta_datos)
        validar_esquema_entrenamiento(df.columns)

        X = df[COLUMNAS_PREDICTORAS].copy()
        y = df[TARGET].copy()
        validar_target(y)

        print("2. Reservando holdout final antes de ajustar transformaciones...")
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y,
        )

        print("3. Ejecutando validacion cruzada fold-safe sobre training data...")
        pipeline = crear_pipeline_baseline()
        cv = StratifiedKFold(
            n_splits=CV_SPLITS,
            shuffle=True,
            random_state=RANDOM_STATE,
        )
        resultados_cv = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=SCORING,
            return_train_score=False,
        )
        metricas_cv = {
            nombre: {
                "media": float(resultados_cv[f"test_{nombre}"].mean()),
                "desviacion": float(resultados_cv[f"test_{nombre}"].std()),
            }
            for nombre in SCORING
        }
        self._imprimir_metricas_cv(metricas_cv)

        print("4. Ajustando el baseline con todo el training set...")
        pipeline.fit(X_train, y_train)

        print("5. Evaluando una unica vez sobre el holdout final...")
        predicciones = pipeline.predict(X_test)
        indice_positivo = list(pipeline.classes_).index(1)
        probabilidades = pipeline.predict_proba(X_test)[:, indice_positivo]
        metricas_holdout = {
            "precision": float(precision_score(y_test, predicciones)),
            "recall": float(recall_score(y_test, predicciones)),
            "f1": float(f1_score(y_test, predicciones)),
            "roc_auc": float(roc_auc_score(y_test, probabilidades)),
            # PR-AUC se reporta como Average Precision.
            "pr_auc": float(average_precision_score(y_test, probabilidades)),
        }
        self._imprimir_metricas_holdout(metricas_holdout)

        print("6. Persistiendo exactamente el pipeline evaluado...")
        self.ruta_pipeline.parent.mkdir(parents=True, exist_ok=True)
        with self.ruta_pipeline.open("wb") as archivo:
            pickle.dump(pipeline, archivo)
        print(f"Pipeline guardado en: {self.ruta_pipeline}")

        return metricas_cv, metricas_holdout

    @staticmethod
    def _imprimir_metricas_cv(metricas):
        print("\n--- Metricas CV (training data, 5 folds) ---")
        for nombre, valores in metricas.items():
            print(
                f"{nombre}: {valores['media']:.6f} "
                f"(+/- {valores['desviacion']:.6f})"
            )

    @staticmethod
    def _imprimir_metricas_holdout(metricas):
        print("\n--- Metricas holdout final ---")
        for nombre, valor in metricas.items():
            print(f"{nombre}: {valor:.6f}")


if __name__ == "__main__":
    entrenador = EntrenadorModelo(
        ruta_datos="data/loan_risk_prediction_dataset.csv",
        ruta_pipeline="artifacts/pipeline_rf_baseline.pkl",
    )
    entrenador.ejecutar()
