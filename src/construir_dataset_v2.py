"""Construccion reproducible del Dataset V2 desde HMDA 2023 oficial.

El modulo consume el CSV del Data Browser como un stream comprimido y aplica
reservoir sampling. El LAR nacional completo nunca se persiste localmente.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import random
import urllib.request
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


RANDOM_STATE = 42
SAMPLE_SIZE = 50_000
TARGET_COLUMN = "LoanApproved"

OFFICIAL_SOURCE_PAGE = "https://ffiec.cfpb.gov/data-publication/2023"
OFFICIAL_API_DOCUMENTATION = "https://ffiec.cfpb.gov/documentation/api/data-browser/"
OFFICIAL_SOURCE_URL = (
    "https://ffiec.cfpb.gov/v2/data-browser-api/view/nationwide/csv"
    "?years=2023&actions_taken=1,2,3&total_units=1,2,3,4"
)
SOURCE_RELEASE = "One Year National Loan-Level Dataset 2023"
SOURCE_FREEZE_DATE = "2025-05-19"

MODEL_PREDICTOR_COLUMNS = (
    "income",
    "loan_amount",
    "loan_term",
    "loan_purpose",
    "loan_type",
    "lien_status",
    "preapproval",
    "debt_to_income_ratio",
    "combined_loan_to_value_ratio",
    "property_value",
    "occupancy_type",
    "construction_method",
    "total_units",
    "submission_of_application",
    "interest_only_payment",
    "balloon_payment",
)

AUDIT_ONLY_COLUMNS = (
    "applicant_age",
    "derived_race",
    "derived_ethnicity",
    "derived_sex",
)

FINAL_COLUMNS = MODEL_PREDICTOR_COLUMNS + AUDIT_ONLY_COLUMNS + (TARGET_COLUMN,)

SOURCE_TO_FINAL = {
    "income": "income",
    "loan_amount": "loan_amount",
    "loan_term": "loan_term",
    "loan_purpose": "loan_purpose",
    "loan_type": "loan_type",
    "lien_status": "lien_status",
    "preapproval": "preapproval",
    "debt_to_income_ratio": "debt_to_income_ratio",
    "loan_to_value_ratio": "combined_loan_to_value_ratio",
    "property_value": "property_value",
    "occupancy_type": "occupancy_type",
    "construction_method": "construction_method",
    "total_units": "total_units",
    "submission_of_application": "submission_of_application",
    "interest_only_payment": "interest_only_payment",
    "balloon_payment": "balloon_payment",
    "applicant_age": "applicant_age",
    "derived_race": "derived_race",
    "derived_ethnicity": "derived_ethnicity",
    "derived_sex": "derived_sex",
}

FILTER_SOURCE_COLUMNS = (
    "activity_year",
    "action_taken",
    "reverse_mortgage",
    "open-end_line_of_credit",
    "business_or_commercial_purpose",
    "total_units",
)

# Lista defensiva: ninguna de estas columnas puede aparecer en el dataset final.
LEAKAGE_BLACKLIST = frozenset(
    {
        "action_taken",
        "lei",
        "derived_msa-md",
        "state_code",
        "county_code",
        "census_tract",
        "purchaser_type",
        "interest_rate",
        "rate_spread",
        "hoepa_status",
        "total_loan_costs",
        "total_points_and_fees",
        "origination_charges",
        "discount_points",
        "lender_credits",
        "prepayment_penalty_term",
        "initially_payable_to_institution",
        "aus-1",
        "aus-2",
        "aus-3",
        "aus-4",
        "aus-5",
        "denial_reason-1",
        "denial_reason-2",
        "denial_reason-3",
        "denial_reason-4",
        "applicant_credit_score_type",
        "co-applicant_credit_score_type",
        "applicant_ethnicity-1",
        "applicant_ethnicity-2",
        "applicant_ethnicity-3",
        "applicant_ethnicity-4",
        "applicant_ethnicity-5",
        "co-applicant_ethnicity-1",
        "co-applicant_ethnicity-2",
        "co-applicant_ethnicity-3",
        "co-applicant_ethnicity-4",
        "co-applicant_ethnicity-5",
        "applicant_race-1",
        "applicant_race-2",
        "applicant_race-3",
        "applicant_race-4",
        "applicant_race-5",
        "co-applicant_race-1",
        "co-applicant_race-2",
        "co-applicant_race-3",
        "co-applicant_race-4",
        "co-applicant_race-5",
        "applicant_sex",
        "co-applicant_sex",
        "applicant_ethnicity_observed",
        "co-applicant_ethnicity_observed",
        "applicant_race_observed",
        "co-applicant_race_observed",
        "applicant_sex_observed",
        "co-applicant_sex_observed",
        "co-applicant_age",
        "applicant_age_above_62",
        "co-applicant_age_above_62",
        "manufactured_home_secured_property_type",
        "manufactured_home_land_property_interest",
        "tract_population",
        "tract_minority_population_percent",
        "ffiec_msa_md_median_family_income",
        "tract_to_msa_income_percentage",
        "tract_owner_occupied_units",
        "tract_one_to_four_family_homes",
        "tract_median_age_of_housing_units",
    }
)

NUMERIC_COLUMNS = (
    "income",
    "loan_amount",
    "loan_term",
    "combined_loan_to_value_ratio",
    "property_value",
)

EXEMPT_CODE_COLUMNS = (
    "preapproval",
    "submission_of_application",
    "interest_only_payment",
    "balloon_payment",
)

MISSING_MARKERS = frozenset({"", "NA", "N/A", "EXEMPT", "NULL", "NONE"})


def _clean_text(value: Any) -> str | None:
    if value is None:
        return None
    cleaned = str(value).strip()
    if cleaned.upper() in MISSING_MARKERS:
        return None
    return cleaned


def _clean_numeric(value: Any) -> float | None:
    cleaned = _clean_text(value)
    if cleaned is None:
        return None
    try:
        return float(cleaned)
    except ValueError as exc:
        raise ValueError(f"Valor numerico HMDA invalido: {value!r}") from exc


def mapear_target(action_taken: Any) -> int:
    """Mapea exclusivamente las tres acciones aprobadas al target binario."""
    action = _clean_text(action_taken)
    if action in {"1", "2"}:
        return 1
    if action == "3":
        return 0
    raise ValueError(f"action_taken fuera del universo binario: {action_taken!r}")


def pertenece_poblacion(row: Mapping[str, Any]) -> bool:
    """Valida los filtros de poblacion aun cuando dos ya operen en servidor."""
    return (
        _clean_text(row.get("activity_year")) == "2023"
        and _clean_text(row.get("action_taken")) in {"1", "2", "3"}
        and _clean_text(row.get("reverse_mortgage")) == "2"
        and _clean_text(row.get("open-end_line_of_credit")) == "2"
        and _clean_text(row.get("business_or_commercial_purpose")) == "2"
        and _clean_text(row.get("total_units")) in {"1", "2", "3", "4"}
    )


def normalizar_registro(row: Mapping[str, Any]) -> dict[str, Any]:
    """Selecciona la whitelist, normaliza faltantes y elimina action_taken."""
    missing = (set(SOURCE_TO_FINAL) | set(FILTER_SOURCE_COLUMNS)) - set(row)
    if missing:
        raise ValueError(f"El esquema HMDA no contiene columnas requeridas: {sorted(missing)}")

    result: dict[str, Any] = {}
    for source, final in SOURCE_TO_FINAL.items():
        value = _clean_text(row[source])
        if final in EXEMPT_CODE_COLUMNS and value == "1111":
            value = None
        if final == "applicant_age" and value == "8888":
            value = None
        result[final] = value

    for column in NUMERIC_COLUMNS:
        result[column] = _clean_numeric(result[column])

    # La fuente publica usa loan_to_value_ratio para el CLTV reportado.
    result[TARGET_COLUMN] = mapear_target(row["action_taken"])
    return result


def muestrear_registros(
    rows: Iterable[Mapping[str, Any]],
    sample_size: int = SAMPLE_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Aplica un reservoir sample uniforme sobre todos los elegibles."""
    if sample_size <= 0:
        raise ValueError("sample_size debe ser positivo")

    rng = random.Random(random_state)
    reservoir: list[dict[str, Any]] = []
    source_rows = 0
    eligible_rows = 0
    eligible_target_counts: Counter[int] = Counter()

    for row in rows:
        source_rows += 1
        if not pertenece_poblacion(row):
            continue
        normalized = normalizar_registro(row)
        eligible_rows += 1
        eligible_target_counts[normalized[TARGET_COLUMN]] += 1

        if len(reservoir) < sample_size:
            reservoir.append(normalized)
            continue
        replacement_index = rng.randrange(eligible_rows)
        if replacement_index < sample_size:
            reservoir[replacement_index] = normalized

    if eligible_rows < sample_size:
        raise ValueError(
            f"Solo existen {eligible_rows} filas elegibles; se solicitaron {sample_size}."
        )

    rng.shuffle(reservoir)
    stats = {
        "source_rows_streamed": source_rows,
        "eligible_population_rows": eligible_rows,
        "eligible_population_target_counts": {
            str(key): eligible_target_counts[key] for key in sorted(eligible_target_counts)
        },
    }
    return reservoir, stats


def iterar_csv_oficial(source_url: str = OFFICIAL_SOURCE_URL) -> Iterator[dict[str, str]]:
    """Lee el endpoint oficial en gzip sin escribir el archivo nacional."""
    request = urllib.request.Request(source_url, headers={"Accept-Encoding": "gzip"})
    with urllib.request.urlopen(request, timeout=180) as response:  # noqa: S310
        raw: Any = response
        if response.headers.get("Content-Encoding", "").lower() == "gzip":
            raw = gzip.GzipFile(fileobj=response)
        with io.TextIOWrapper(raw, encoding="utf-8-sig", newline="") as text_stream:
            reader = csv.DictReader(text_stream)
            if reader.fieldnames is None:
                raise ValueError("La fuente oficial no devolvio encabezados CSV.")
            required = set(SOURCE_TO_FINAL) | set(FILTER_SOURCE_COLUMNS)
            missing = required - set(reader.fieldnames)
            if missing:
                raise ValueError(
                    f"La fuente oficial cambio de esquema; faltan: {sorted(missing)}"
                )
            yield from reader


def obtener_predictores_entrenamiento(data: pd.DataFrame) -> pd.DataFrame:
    """Guardrail: devuelve solo las 16 variables autorizadas para modelado."""
    missing = set(MODEL_PREDICTOR_COLUMNS) - set(data.columns)
    if missing:
        raise ValueError(f"Faltan predictores V2 requeridos: {sorted(missing)}")
    predictors = data.loc[:, MODEL_PREDICTOR_COLUMNS].copy()
    forbidden = set(predictors.columns) & (set(AUDIT_ONLY_COLUMNS) | {TARGET_COLUMN})
    if forbidden:
        raise AssertionError(f"Columnas no predictoras filtradas incorrectamente: {forbidden}")
    return predictors


def validar_dataset(data: pd.DataFrame, expected_rows: int = SAMPLE_SIZE) -> None:
    """Valida el contrato final de esquema, roles y target."""
    if data.columns.duplicated().any():
        raise ValueError("El Dataset V2 contiene nombres de columna duplicados.")
    if tuple(data.columns) != FINAL_COLUMNS:
        raise ValueError(
            "Esquema Dataset V2 invalido. "
            f"Esperado {list(FINAL_COLUMNS)}, recibido {list(data.columns)}."
        )
    if len(data) != expected_rows:
        raise ValueError(f"Se esperaban {expected_rows} filas y se obtuvieron {len(data)}.")
    if data[TARGET_COLUMN].isna().any():
        raise ValueError("LoanApproved no admite valores faltantes.")
    if set(data[TARGET_COLUMN].unique()) != {0, 1}:
        raise ValueError("LoanApproved debe contener exclusivamente las clases {0, 1}.")
    leaked = set(data.columns) & LEAKAGE_BLACKLIST
    if leaked:
        raise ValueError(f"El Dataset V2 contiene columnas prohibidas: {sorted(leaked)}")
    obtener_predictores_entrenamiento(data)


def _missingness(data: pd.DataFrame) -> dict[str, float]:
    return {
        column: round(float(data[column].isna().mean()), 6)
        for column in data.columns
    }


def crear_metadata(
    data: pd.DataFrame,
    source_stats: Mapping[str, Any],
    output_path: Path,
    extracted_at: str,
) -> dict[str, Any]:
    target_counts = data[TARGET_COLUMN].value_counts().sort_index()
    by_target = {
        str(target): _missingness(group)
        for target, group in data.groupby(TARGET_COLUMN, sort=True)
    }
    return {
        "dataset": "HMDA 2023 Loan Approval V2",
        "source": {
            "publisher": "CFPB/FFIEC",
            "release": SOURCE_RELEASE,
            "data_freeze_date": SOURCE_FREEZE_DATE,
            "official_publication_page": OFFICIAL_SOURCE_PAGE,
            "api_documentation": OFFICIAL_API_DOCUMENTATION,
            "exact_query": OFFICIAL_SOURCE_URL,
            "retrieved_at_utc": extracted_at,
            "raw_national_file_persisted": False,
            "stream_compression": "gzip",
        },
        "population_filters": {
            "activity_year": 2023,
            "action_taken": [1, 2, 3],
            "reverse_mortgage": 2,
            "open-end_line_of_credit": 2,
            "business_or_commercial_purpose": 2,
            "total_units": [1, 2, 3, 4],
        },
        "target_mapping": {"1": 1, "2": 1, "3": 0},
        "sampling": {
            "method": "uniform reservoir sampling after population filters",
            "random_state": RANDOM_STATE,
            "requested_rows": SAMPLE_SIZE,
            "artificial_class_balancing": False,
            **dict(source_stats),
        },
        "shape": {"rows": len(data), "columns": len(data.columns)},
        "target_distribution": {
            str(target): {
                "count": int(count),
                "proportion": round(float(count / len(data)), 6),
            }
            for target, count in target_counts.items()
        },
        "missingness_global": _missingness(data),
        "missingness_by_target": by_target,
        "column_roles": {
            "predictor": list(MODEL_PREDICTOR_COLUMNS),
            "audit_only": list(AUDIT_ONLY_COLUMNS),
            "target": [TARGET_COLUMN],
        },
        "schema_decisions": {
            "removed_after_audit": {
                "negative_amortization": "constant in the 50,000-row sample",
                "introductory_rate_period": "93.214% structural missingness",
                "other_nonamortizing_features": "44 positive values (0.088%)",
            },
            "retained_after_audit": {
                "total_units": "868 multi-unit rows and four observed categories",
            },
        },
        "output": {
            "path": output_path.as_posix(),
            "size_bytes": output_path.stat().st_size,
            "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        },
    }


def construir_dataset(
    output_path: Path,
    metadata_path: Path,
    source_url: str = OFFICIAL_SOURCE_URL,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    extracted_at = datetime.now(timezone.utc).isoformat()
    sample, source_stats = muestrear_registros(iterar_csv_oficial(source_url))
    data = pd.DataFrame.from_records(sample, columns=FINAL_COLUMNS)
    data[TARGET_COLUMN] = data[TARGET_COLUMN].astype("int8")
    validar_dataset(data)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False, lineterminator="\n")
    if output_path.stat().st_size >= 100 * 1024 * 1024:
        raise ValueError("El CSV final supera el limite aprobado de 100 MiB.")

    metadata = crear_metadata(data, source_stats, output_path, extracted_at)
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return data, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/hmda_2023_loan_approval_v2.csv"),
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("data/hmda_2023_loan_approval_v2.metadata.json"),
    )
    args = parser.parse_args()
    data, metadata = construir_dataset(args.output, args.metadata)
    print(f"Dataset V2 generado: {data.shape[0]} filas x {data.shape[1]} columnas")
    print(json.dumps(metadata["target_distribution"], indent=2))


if __name__ == "__main__":
    main()
