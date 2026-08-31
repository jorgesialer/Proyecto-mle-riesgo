"""Construccion del pipeline reproducible para el baseline Random Forest."""

from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder


TARGET = "LoanApproved"
COLUMNAS_NUMERICAS_INVALIDABLES = ["Income", "LoanAmount"]
COLUMNAS_NUMERICAS = ["Age", "CreditScore", "YearsExperience"]
COLUMNAS_CATEGORICAS = ["Gender", "Education", "City", "EmploymentType"]
COLUMNAS_PREDICTORAS = (
    COLUMNAS_NUMERICAS_INVALIDABLES + COLUMNAS_NUMERICAS + COLUMNAS_CATEGORICAS
)


def invalidar_valores_negativos(valores):
    """Convierte valores negativos en NaN sin aprender parametros de los datos."""
    arreglo = np.asarray(valores, dtype=float).copy()
    arreglo[arreglo < 0] = np.nan
    return arreglo


def crear_preprocesador() -> ColumnTransformer:
    """Crea el preprocesamiento que se ajustara exclusivamente con training data."""
    numericas_invalidables = Pipeline(
        steps=[
            (
                "invalidar_negativos",
                FunctionTransformer(
                    invalidar_valores_negativos,
                    validate=False,
                    feature_names_out="one-to-one",
                ),
            ),
            ("imputar", SimpleImputer(strategy="median")),
        ]
    )

    numericas = Pipeline(
        steps=[("imputar", SimpleImputer(strategy="median"))]
    )

    categoricas = Pipeline(
        steps=[
            ("imputar", SimpleImputer(strategy="most_frequent")),
            (
                "codificar",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "numericas_invalidables",
                numericas_invalidables,
                COLUMNAS_NUMERICAS_INVALIDABLES,
            ),
            ("numericas", numericas, COLUMNAS_NUMERICAS),
            ("categoricas", categoricas, COLUMNAS_CATEGORICAS),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def crear_pipeline_baseline() -> Pipeline:
    """Crea el artefacto completo de preprocesamiento y clasificacion."""
    return Pipeline(
        steps=[
            ("preprocesamiento", crear_preprocesador()),
            (
                "clasificador",
                RandomForestClassifier(
                    n_estimators=100,
                    random_state=42,
                    class_weight="balanced",
                ),
            ),
        ]
    )


def validar_columnas_predictoras(columnas) -> None:
    """Exige exactamente el contrato de predictores crudos del baseline."""
    lista_columnas = list(columnas)
    if len(lista_columnas) != len(set(lista_columnas)):
        raise ValueError("Contrato de predictores invalido: hay columnas duplicadas")

    columnas_disponibles = set(lista_columnas)
    columnas_esperadas = set(COLUMNAS_PREDICTORAS)
    faltantes = sorted(columnas_esperadas - columnas_disponibles)
    inesperadas = sorted(columnas_disponibles - columnas_esperadas)

    errores = []
    if faltantes:
        errores.append(f"faltan columnas: {faltantes}")
    if inesperadas:
        errores.append(f"columnas inesperadas: {inesperadas}")
    if errores:
        raise ValueError("Contrato de predictores invalido: " + "; ".join(errores))


def validar_esquema_entrenamiento(columnas) -> None:
    """Valida el target y los predictores del dataset crudo de entrenamiento."""
    lista_columnas = list(columnas)
    if lista_columnas.count(TARGET) != 1:
        raise ValueError(f"El dataset debe contener exactamente una columna {TARGET}")
    validar_columnas_predictoras([col for col in lista_columnas if col != TARGET])
