"""RAG V2 reproducible sobre fuentes oficiales HMDA y mortgage underwriting.

La capa es deliberadamente independiente de cualquier LLM: adquiere un corpus
versionado, extrae y fragmenta texto, crea embeddings locales, persiste un
indice Qdrant embedded y evalua retrieval con juicios manuales transparentes.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import shutil
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any, Iterable, Sequence
from urllib.parse import urlparse

import mlflow
import numpy as np
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from pypdf import PdfReader
from qdrant_client import QdrantClient, models

from src.construir_dataset_v2 import AUDIT_ONLY_COLUMNS
from src.entrenamiento_v2 import EXPERIMENT_NAME
from src.mlflow_utils import TrackingConfig, configurar_mlflow, validar_run_name


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DIR = PROJECT_ROOT / "data" / "rag_sources"
DEFAULT_METADATA_PATH = DEFAULT_SOURCE_DIR / "corpus_metadata.json"
DEFAULT_EVAL_PATH = PROJECT_ROOT / "data" / "rag_eval_queries.json"
DEFAULT_QDRANT_PATH = PROJECT_ROOT / "data" / "qdrant" / "credit_policy_v2_v2"
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "rag"
COLLECTION_NAME = "credit_policy_v2"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384
CHUNK_SIZE_CHARS = 1_000
CHUNK_OVERLAP_CHARS = 150
DEFAULT_TOP_K = 5
RAG_RUN_NAME = "rag_qdrant_retrieval_v2"

OFFICIAL_HOSTS = frozenset(
    {
        "files.consumerfinance.gov",
        "selling-guide.fanniemae.com",
    }
)


@dataclass(frozen=True)
class CorpusSource:
    source_id: str
    title: str
    institution: str
    source_url: str
    document_date: str
    document_version: str | None
    local_filename: str
    media_type: str


CORPUS_SOURCES = (
    CorpusSource(
        source_id="cfpb_hmda_2023_reference",
        title="Reportable HMDA Data: 2023 Regulatory and Reporting Overview Reference Chart",
        institution="Consumer Financial Protection Bureau (CFPB)",
        source_url=(
            "https://files.consumerfinance.gov/f/documents/"
            "cfpb_reportable-hmda-data_regulatory-and-reporting-overview-"
            "reference-chart_2023-02.pdf"
        ),
        document_date="2023-02-09",
        document_version="Version 1.0",
        local_filename="cfpb_hmda_2023_reference_chart.pdf",
        media_type="application/pdf",
    ),
    CorpusSource(
        source_id="fannie_general_income",
        title="B3-3.1-01, General Income Information",
        institution="Fannie Mae",
        source_url=(
            "https://selling-guide.fanniemae.com/sel/b3-3.1-01/"
            "general-income-information"
        ),
        document_date="2026-03-04",
        document_version="Selling Guide section B3-3.1-01",
        local_filename="fannie_b3_3_1_01_general_income.html",
        media_type="text/html",
    ),
    CorpusSource(
        source_id="fannie_debt_to_income",
        title="B3-6-02, Debt-to-Income Ratios",
        institution="Fannie Mae",
        source_url=(
            "https://selling-guide.fanniemae.com/sel/b3-6-02/"
            "debt-income-ratios"
        ),
        document_date="2025-04-02",
        document_version="Selling Guide section B3-6-02",
        local_filename="fannie_b3_6_02_debt_to_income.html",
        media_type="text/html",
    ),
    CorpusSource(
        source_id="fannie_ltv",
        title="B2-1.2-01, Loan-to-Value (LTV) Ratios",
        institution="Fannie Mae",
        source_url=(
            "https://selling-guide.fanniemae.com/sel/b2-1.2-01/"
            "loan-value-ltv-ratios"
        ),
        document_date="2022-06-01",
        document_version="Selling Guide section B2-1.2-01",
        local_filename="fannie_b2_1_2_01_ltv.html",
        media_type="text/html",
    ),
    CorpusSource(
        source_id="fannie_cltv",
        title="B2-1.2-02, Combined Loan-to-Value (CLTV) Ratios",
        institution="Fannie Mae",
        source_url=(
            "https://selling-guide.fanniemae.com/sel/b2-1.2-02/"
            "combined-loan-value-cltv-ratios"
        ),
        document_date="2018-12-04",
        document_version="Selling Guide section B2-1.2-02",
        local_filename="fannie_b2_1_2_02_cltv.html",
        media_type="text/html",
    ),
)


@dataclass(frozen=True)
class TextSection:
    source_id: str
    title: str
    institution: str
    source_url: str
    page_or_section: str
    text: str


@dataclass(frozen=True)
class TextChunk:
    chunk_id: str
    source_id: str
    source_title: str
    institution: str
    source_url: str
    page_or_section: str
    text: str

    def embedding_text(self) -> str:
        return f"{self.source_title}\n{self.page_or_section}\n{self.text}"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validar_fuente_oficial(source: CorpusSource) -> None:
    parsed = urlparse(source.source_url)
    if parsed.scheme != "https" or parsed.hostname not in OFFICIAL_HOSTS:
        raise ValueError(f"Fuente no oficial o URL no HTTPS: {source.source_url}")
    if not source.local_filename.endswith((".pdf", ".html")):
        raise ValueError(f"Formato de corpus no permitido: {source.local_filename}")


def adquirir_corpus(
    source_dir: Path = DEFAULT_SOURCE_DIR,
    metadata_path: Path = DEFAULT_METADATA_PATH,
    *,
    force: bool = False,
    timeout_seconds: int = 60,
) -> list[dict[str, Any]]:
    """Descarga snapshots oficiales y registra trazabilidad criptografica."""
    source_dir.mkdir(parents=True, exist_ok=True)
    existing: dict[str, dict[str, Any]] = {}
    if metadata_path.exists():
        existing = {
            item["source_id"]: item
            for item in json.loads(metadata_path.read_text(encoding="utf-8"))["documents"]
        }

    documents: list[dict[str, Any]] = []
    headers = {"User-Agent": "Proyecto-MLE-Riesgo/2.0 educational RAG corpus"}
    for source in CORPUS_SOURCES:
        validar_fuente_oficial(source)
        local_path = source_dir / source.local_filename
        prior = existing.get(source.source_id)
        if not force and local_path.exists() and prior:
            actual_hash = _sha256(local_path)
            if actual_hash != prior.get("sha256"):
                raise ValueError(
                    f"Hash local inesperado para {source.source_id}; use force=True "
                    "solo para adquirir una nueva revision conscientemente"
                )
            documents.append(prior)
            continue

        response = requests.get(
            source.source_url,
            headers=headers,
            timeout=timeout_seconds,
        )
        response.raise_for_status()
        content_type = response.headers.get("content-type", "").lower()
        expected_family = source.media_type.split("/")[1]
        if expected_family not in content_type:
            raise ValueError(
                f"Content-Type inesperado para {source.source_id}: {content_type!r}"
            )
        local_path.write_bytes(response.content)
        item = asdict(source) | {
            "retrieved_at": datetime.now(UTC).isoformat(),
            "sha256": _sha256(local_path),
            "bytes": local_path.stat().st_size,
        }
        documents.append(item)

    manifest = {
        "corpus_id": "credit_policy_v2_official_v1",
        "document_count": len(documents),
        "official_hosts": sorted(OFFICIAL_HOSTS),
        "documents": documents,
    }
    _write_json(metadata_path, manifest)
    return documents


def _clean_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def extraer_pdf(path: Path, source: CorpusSource) -> list[TextSection]:
    sections: list[TextSection] = []
    reader = PdfReader(path)
    for page_number, page in enumerate(reader.pages, start=1):
        text = _clean_text(page.extract_text() or "")
        if text:
            sections.append(
                TextSection(
                    source_id=source.source_id,
                    title=source.title,
                    institution=source.institution,
                    source_url=source.source_url,
                    page_or_section=f"page {page_number}",
                    text=text,
                )
            )
    if not sections:
        raise ValueError(f"No se extrajo texto de {path}")
    return sections


def extraer_html(path: Path, source: CorpusSource) -> list[TextSection]:
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="replace"), "html.parser")
    for node in soup.select("script, style, nav, header, footer, form, noscript, svg"):
        node.decompose()
    root = soup.select_one("main, article, [role='main']") or soup.body or soup
    sections: list[TextSection] = []
    current_heading = source.title
    current_parts: list[str] = []

    def flush() -> None:
        text = _clean_text("\n".join(current_parts))
        if text:
            sections.append(
                TextSection(
                    source_id=source.source_id,
                    title=source.title,
                    institution=source.institution,
                    source_url=source.source_url,
                    page_or_section=current_heading,
                    text=text,
                )
            )

    for node in root.find_all(["h1", "h2", "h3", "h4", "p", "li"]):
        if node.name == "li" and node.find(["p", "li"]) is not None:
            # El contenido de los bloques descendientes se procesara cuando
            # llegue su turno. Solo se conserva texto directo adicional del
            # <li>, evitando duplicar casos como <li><p>texto</p></li>.
            text = _clean_text(
                " ".join(str(value) for value in node.find_all(string=True, recursive=False))
            )
        else:
            text = _clean_text(node.get_text(" ", strip=True))
        if not text:
            continue
        if node.name in {"h1", "h2", "h3", "h4"}:
            flush()
            current_parts = []
            current_heading = text
        else:
            current_parts.append(text)
    flush()
    if not sections:
        raise ValueError(f"No se extrajo contenido estructurado de {path}")
    return sections


def extraer_corpus(
    source_dir: Path = DEFAULT_SOURCE_DIR,
) -> list[TextSection]:
    sections: list[TextSection] = []
    for source in CORPUS_SOURCES:
        path = source_dir / source.local_filename
        if not path.exists():
            raise FileNotFoundError(
                f"Falta {path}; ejecute adquirir_corpus antes de extraer"
            )
        if source.media_type == "application/pdf":
            sections.extend(extraer_pdf(path, source))
        else:
            sections.extend(extraer_html(path, source))
    return sections


def _split_words(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    if max_chars <= 0 or overlap_chars < 0 or overlap_chars >= max_chars:
        raise ValueError("Configuracion de chunking invalida")
    words = text.split()
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = start
        length = 0
        while end < len(words):
            candidate_length = length + (1 if length else 0) + len(words[end])
            if candidate_length > max_chars and end > start:
                break
            length = candidate_length
            end += 1
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        overlap_length = 0
        next_start = end
        while next_start > start and overlap_length < overlap_chars:
            next_start -= 1
            overlap_length += len(words[next_start]) + 1
        start = max(next_start, start + 1)
    return chunks


def crear_chunks(
    sections: Iterable[TextSection],
    *,
    chunk_size_chars: int = CHUNK_SIZE_CHARS,
    chunk_overlap_chars: int = CHUNK_OVERLAP_CHARS,
) -> list[TextChunk]:
    chunks: list[TextChunk] = []
    counters: dict[str, int] = {}
    for section in sections:
        for text in _split_words(
            section.text,
            max_chars=chunk_size_chars,
            overlap_chars=chunk_overlap_chars,
        ):
            index = counters.get(section.source_id, 0)
            counters[section.source_id] = index + 1
            chunks.append(
                TextChunk(
                    chunk_id=f"{section.source_id}:{index:04d}",
                    source_id=section.source_id,
                    source_title=section.title,
                    institution=section.institution,
                    source_url=section.source_url,
                    page_or_section=section.page_or_section,
                    text=text,
                )
            )
    if not chunks:
        raise ValueError("El corpus no produjo chunks")
    return chunks


class LocalSentenceEmbedder:
    """Wrapper pequeno para embeddings locales normalizados."""

    def __init__(self, model_name: str = EMBEDDING_MODEL) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._model = SentenceTransformer(model_name, device="cpu")
        self.dimension = int(self._model.get_embedding_dimension())
        if model_name == EMBEDDING_MODEL and self.dimension != EMBEDDING_DIMENSION:
            raise ValueError(
                f"Dimension inesperada para {model_name}: {self.dimension}"
            )

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        values = self._model.encode(
            list(texts),
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return np.asarray(values, dtype=np.float32)


def _point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"credit-policy-v2/{chunk_id}"))


def build_index(
    chunks: Sequence[TextChunk],
    *,
    qdrant_path: Path = DEFAULT_QDRANT_PATH,
    collection_name: str = COLLECTION_NAME,
    embedder: Any | None = None,
    client: QdrantClient | None = None,
) -> dict[str, Any]:
    """Crea de cero una coleccion Qdrant local a partir de chunks trazables."""
    embedder = embedder or LocalSentenceEmbedder()
    owns_client = client is None
    if client is None:
        qdrant_path.mkdir(parents=True, exist_ok=True)
        client = QdrantClient(path=str(qdrant_path))
    try:
        if client.collection_exists(collection_name):
            existing_count = client.count(collection_name=collection_name, exact=True).count
            existing_signatures: dict[str, str] = {}
            offset = None
            while True:
                records, offset = client.scroll(
                    collection_name=collection_name,
                    limit=256,
                    offset=offset,
                    with_payload=["chunk_id", "text"],
                    with_vectors=False,
                )
                for record in records:
                    if record.payload:
                        chunk_id = str(record.payload.get("chunk_id"))
                        chunk_text = str(record.payload.get("text"))
                        existing_signatures[chunk_id] = hashlib.sha256(
                            chunk_text.encode("utf-8")
                        ).hexdigest()
                if offset is None:
                    break
            expected_signatures = {
                chunk.chunk_id: hashlib.sha256(chunk.text.encode("utf-8")).hexdigest()
                for chunk in chunks
            }
            if (
                existing_count == len(chunks)
                and existing_signatures == expected_signatures
            ):
                return {
                    "collection_name": collection_name,
                    "storage_mode": "embedded_local",
                    "distance": "Cosine",
                    "vector_dimension": int(embedder.dimension),
                    "point_count": int(existing_count),
                    "embedding_model": getattr(
                        embedder, "model_name", type(embedder).__name__
                    ),
                }
            client.delete_collection(collection_name)
            # Qdrant local persiste la eliminacion al cerrar el cliente. Reabrir
            # evita que una recreacion inmediata reutilice puntos del snapshot
            # anterior en ciertos sistemas de archivos sincronizados.
            if owns_client:
                client.close()
                del client
                gc.collect()
                collection_root = (qdrant_path / "collection").resolve()
                collection_storage = (collection_root / collection_name).resolve()
                if collection_storage.parent != collection_root:
                    raise ValueError("Nombre de coleccion Qdrant inseguro")
                # Qdrant local puede dejar el directorio de una coleccion
                # eliminada en sistemas sincronizados (p. ej. OneDrive). Solo
                # se retira ese almacenamiento interno ya desregistrado.
                if collection_storage.exists():
                    shutil.rmtree(collection_storage)
                client = QdrantClient(path=str(qdrant_path))
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=int(embedder.dimension),
                distance=models.Distance.COSINE,
            ),
        )
        initial_count = client.count(collection_name=collection_name, exact=True).count
        if initial_count != 0:
            raise RuntimeError(
                f"La coleccion recreada no esta vacia: {initial_count} puntos"
            )
        embeddings = embedder.encode([chunk.embedding_text() for chunk in chunks])
        if embeddings.shape != (len(chunks), int(embedder.dimension)):
            raise ValueError(f"Forma de embeddings inesperada: {embeddings.shape}")
        points = [
            models.PointStruct(
                id=_point_id(chunk.chunk_id),
                vector=embeddings[index].tolist(),
                payload=asdict(chunk),
            )
            for index, chunk in enumerate(chunks)
        ]
        for start in range(0, len(points), 128):
            client.upsert(
                collection_name=collection_name,
                points=points[start : start + 128],
                wait=True,
            )
        count = client.count(collection_name=collection_name, exact=True).count
        if count != len(chunks):
            raise RuntimeError(f"Qdrant contiene {count} puntos; se esperaban {len(chunks)}")
        return {
            "collection_name": collection_name,
            "storage_mode": "embedded_local",
            "distance": "Cosine",
            "vector_dimension": int(embedder.dimension),
            "point_count": int(count),
            "embedding_model": getattr(embedder, "model_name", type(embedder).__name__),
        }
    finally:
        if owns_client:
            client.close()


def retrieve_policy_context(
    query: str,
    *,
    k: int = DEFAULT_TOP_K,
    qdrant_path: Path = DEFAULT_QDRANT_PATH,
    collection_name: str = COLLECTION_NAME,
    embedder: Any | None = None,
    client: QdrantClient | None = None,
) -> dict[str, Any]:
    """Recupera contexto normativo con un contrato JSON-friendly."""
    query = query.strip()
    if not query:
        raise ValueError("query no puede estar vacia")
    if k < 1:
        raise ValueError("k debe ser positivo")
    embedder = embedder or LocalSentenceEmbedder()
    owns_client = client is None
    if client is None:
        client = QdrantClient(path=str(qdrant_path))
    try:
        if not client.collection_exists(collection_name):
            raise FileNotFoundError(f"No existe la coleccion Qdrant {collection_name!r}")
        vector = embedder.encode([query])[0].tolist()
        response = client.query_points(
            collection_name=collection_name,
            query=vector,
            limit=k,
            with_payload=True,
            with_vectors=False,
        )
        results = []
        for point in response.points:
            payload = point.payload or {}
            results.append(
                {
                    "text": str(payload["text"]),
                    "score": float(point.score),
                    "source_id": str(payload["source_id"]),
                    "source_title": str(payload["source_title"]),
                    "institution": str(payload["institution"]),
                    "source_url": str(payload["source_url"]),
                    "page_or_section": str(payload["page_or_section"]),
                    "chunk_id": str(payload["chunk_id"]),
                }
            )
        payload = {"query": query, "results": results}
        json.dumps(payload, allow_nan=False)
        return payload
    finally:
        if owns_client:
            client.close()


RAG_QUERY_TEMPLATES = {
    "debt_to_income_ratio": "mortgage underwriting debt-to-income ratio calculation and limits",
    "dti_category": "mortgage underwriting debt-to-income ratio calculation and limits",
    "combined_loan_to_value_ratio": "mortgage combined loan-to-value CLTV calculation and subordinate financing",
    "loan_to_property_value": "mortgage loan-to-value LTV calculation",
    "income": "mortgage underwriting stable predictable income documentation and continuance",
    "loan_to_income": "mortgage underwriting income and debt obligations",
    "property_value_to_income": "mortgage underwriting property value and income",
    "loan_purpose": "HMDA loan purpose definitions and reporting codes",
    "occupancy_type": "HMDA occupancy type principal residence reporting codes",
    "loan_type": "HMDA loan type conventional FHA VA reporting codes",
    "lien_status": "HMDA lien status first lien subordinate lien reporting codes",
    "construction_method": "HMDA construction method site-built manufactured home reporting codes",
    "loan_amount": "HMDA loan amount reporting requirements",
}


def construir_queries_desde_evidence(
    evidence_package: dict[str, Any],
    *,
    max_queries: int = 5,
) -> dict[str, Any]:
    """Traduce factores XAI a queries deterministas; no usa ni simula un LLM."""
    if max_queries < 1:
        raise ValueError("max_queries debe ser positivo")
    forbidden = set(AUDIT_ONLY_COLUMNS)
    factors = list(evidence_package.get("top_positive_factors") or []) + list(
        evidence_package.get("top_negative_factors") or []
    )
    queries: list[dict[str, str]] = []
    seen: set[str] = set()
    for factor in factors:
        model_feature = str(factor.get("model_feature", ""))
        source_features = {str(value) for value in factor.get("source_features", [])}
        if model_feature in forbidden or source_features & forbidden:
            continue
        query = RAG_QUERY_TEMPLATES.get(model_feature)
        if query is None:
            query = next(
                (RAG_QUERY_TEMPLATES[name] for name in source_features if name in RAG_QUERY_TEMPLATES),
                None,
            )
        if query and query not in seen:
            queries.append({"origin_feature": model_feature, "query": query})
            seen.add(query)
        if len(queries) == max_queries:
            break
    payload = {"queries": queries, "generator": "deterministic_feature_template_v1"}
    serialized = json.dumps(payload, allow_nan=False)
    if any(column in serialized for column in AUDIT_ONLY_COLUMNS):
        raise AssertionError("Una columna audit-only alcanzo las queries RAG")
    return payload


def evaluar_retrieval(
    evaluation_set: Sequence[dict[str, Any]],
    *,
    k: int = DEFAULT_TOP_K,
    qdrant_path: Path = DEFAULT_QDRANT_PATH,
    collection_name: str = COLLECTION_NAME,
    embedder: Any | None = None,
    client: QdrantClient | None = None,
) -> dict[str, Any]:
    """Evalua retrieval separando queries dirigidas, adicionales y OOD."""
    embedder = embedder or LocalSentenceEmbedder()
    evaluations = []
    for item in evaluation_set:
        retrieval = retrieve_policy_context(
            item["query"],
            k=k,
            qdrant_path=qdrant_path,
            collection_name=collection_name,
            embedder=embedder,
            client=client,
        )
        expected_sources = set(item.get("expected_source_ids", []))
        expected_no_relevant_source = bool(
            item.get("expected_no_relevant_source", False)
        )
        rank = next(
            (
                index
                for index, result in enumerate(retrieval["results"], start=1)
                if result["source_id"] in expected_sources
            ),
            None,
        )
        combined_text = " ".join(result["text"] for result in retrieval["results"]).lower()
        expected_concepts = [
            str(value).lower() for value in item.get("expected_concepts", [])
        ]
        top_score = (
            float(retrieval["results"][0]["score"])
            if retrieval["results"]
            else None
        )
        if expected_no_relevant_source:
            concept_hit = None
            source_hit = None
            source_recall = None
            reciprocal_rank = None
        else:
            concept_hit = all(
                concept in combined_text for concept in expected_concepts
            )
            retrieved_sources = {
                result["source_id"] for result in retrieval["results"]
            }
            source_hit = rank is not None
            source_recall = (
                len(expected_sources & retrieved_sources) / len(expected_sources)
                if expected_sources
                else 0.0
            )
            reciprocal_rank = 0.0 if rank is None else 1.0 / rank
        evaluations.append(
            {
                "query_id": item["query_id"],
                "query": item["query"],
                "evaluation_group": item.get("evaluation_group", "directed"),
                "query_type": item.get("query_type", "directed"),
                "expected_no_relevant_source": expected_no_relevant_source,
                "expected_source_ids": sorted(expected_sources),
                "expected_concepts": expected_concepts,
                "source_hit_at_k": source_hit,
                "source_recall_at_k": source_recall,
                "concept_hit_at_k": concept_hit,
                "first_relevant_rank": rank,
                "reciprocal_rank": reciprocal_rank,
                "top_score": top_score,
                "result_count": len(retrieval["results"]),
                "rejection_applied": False,
                "retrieved": retrieval["results"],
            }
        )

    def in_domain_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
        if not rows:
            return {"query_count": 0}
        return {
            f"source_hit_at_{k}": float(
                np.mean([float(row["source_hit_at_k"]) for row in rows])
            ),
            f"source_recall_at_{k}": float(
                np.mean([float(row["source_recall_at_k"]) for row in rows])
            ),
            f"concept_hit_at_{k}": float(
                np.mean([float(row["concept_hit_at_k"]) for row in rows])
            ),
            f"mrr_at_{k}": float(
                np.mean([float(row["reciprocal_rank"]) for row in rows])
            ),
            "mean_top_score": float(
                np.mean([float(row["top_score"]) for row in rows])
            ),
            "min_top_score": float(
                np.min([float(row["top_score"]) for row in rows])
            ),
            "query_count": len(rows),
        }

    directed = [
        row
        for row in evaluations
        if row["evaluation_group"] == "directed"
        and not row["expected_no_relevant_source"]
    ]
    additional_in_domain = [
        row
        for row in evaluations
        if row["evaluation_group"] == "additional"
        and not row["expected_no_relevant_source"]
    ]
    all_in_domain = [
        row for row in evaluations if not row["expected_no_relevant_source"]
    ]
    out_of_domain = [
        row for row in evaluations if row["expected_no_relevant_source"]
    ]
    ood_scores = [
        float(row["top_score"])
        for row in out_of_domain
        if row["top_score"] is not None
    ]
    ood_metrics = {
        "query_count": len(out_of_domain),
        "returned_results_count": sum(
            int(row["result_count"] > 0) for row in out_of_domain
        ),
        "rejected_count": sum(
            int(row["rejection_applied"]) for row in out_of_domain
        ),
        "avoidance_rate": float(
            np.mean([float(row["result_count"] == 0) for row in out_of_domain])
        )
        if out_of_domain
        else 0.0,
        "mean_top_score": float(np.mean(ood_scores)) if ood_scores else None,
        "max_top_score": float(np.max(ood_scores)) if ood_scores else None,
    }
    metrics = {
        "directed": in_domain_metrics(directed),
        "additional_in_domain": in_domain_metrics(additional_in_domain),
        "all_in_domain": in_domain_metrics(all_in_domain),
        "out_of_domain": ood_metrics,
    }
    payload = {"k": k, "metrics": metrics, "queries": evaluations}
    json.dumps(payload, allow_nan=False)
    return payload


def _save_chunks(path: Path, chunks: Sequence[TextChunk]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for chunk in chunks:
            file.write(json.dumps(asdict(chunk), ensure_ascii=False, allow_nan=False) + "\n")


def registrar_run_rag(
    tracking: TrackingConfig,
    artifact_dir: Path,
    corpus_metadata_path: Path,
    evaluation_path: Path,
    index_metadata: dict[str, Any],
    evaluation: dict[str, Any],
    *,
    run_name: str = RAG_RUN_NAME,
) -> dict[str, Any]:
    validar_run_name(run_name)
    with mlflow.start_run(experiment_id=tracking.experiment_id, run_name=run_name) as run:
        mlflow.set_tags(
            {
                "stage": "rag_retrieval",
                "corpus": "official_hmda_mortgage_policy",
                "llm_used": "false",
                "qdrant_mode": "embedded_local",
            }
        )
        mlflow.log_params(
            {
                "embedding_model": index_metadata["embedding_model"],
                "embedding_dimension": index_metadata["vector_dimension"],
                "chunk_size_chars": CHUNK_SIZE_CHARS,
                "chunk_overlap_chars": CHUNK_OVERLAP_CHARS,
                "collection_name": index_metadata["collection_name"],
                "document_count": len(CORPUS_SOURCES),
                "chunk_count": index_metadata["point_count"],
                "top_k": evaluation["k"],
            }
        )
        mlflow_metrics = {}
        for group, group_metrics in evaluation["metrics"].items():
            for name, value in group_metrics.items():
                if value is not None:
                    mlflow_metrics[f"{group}_{name}"] = float(value)
        mlflow.log_metrics(mlflow_metrics)
        mlflow.log_artifact(str(corpus_metadata_path), artifact_path="corpus")
        mlflow.log_artifact(str(evaluation_path), artifact_path="evaluation")
        run_metadata = {
            "run_id": run.info.run_id,
            "run_name": run_name,
            "backend": tracking.backend,
            "experiment_name": tracking.experiment_name,
            "tracking_ui": tracking.ui_url,
        }
        _write_json(artifact_dir / "run_metadata.json", run_metadata)
        mlflow.log_artifacts(str(artifact_dir), artifact_path="rag")
        return run_metadata


def ejecutar_rag(
    *,
    backend: str | None = None,
    log_mlflow: bool = True,
    force_download: bool = False,
) -> dict[str, Any]:
    load_dotenv(PROJECT_ROOT / ".env")
    documents = adquirir_corpus(force=force_download)
    sections = extraer_corpus()
    chunks = crear_chunks(sections)
    DEFAULT_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    chunks_path = DEFAULT_ARTIFACT_DIR / "chunks.jsonl"
    _save_chunks(chunks_path, chunks)

    embedder = LocalSentenceEmbedder()
    index_metadata = build_index(chunks, embedder=embedder)
    index_metadata |= {
        "chunk_size_chars": CHUNK_SIZE_CHARS,
        "chunk_overlap_chars": CHUNK_OVERLAP_CHARS,
        "document_count": len(documents),
        "section_count": len(sections),
    }
    _write_json(DEFAULT_ARTIFACT_DIR / "index_metadata.json", index_metadata)

    evaluation_set = json.loads(DEFAULT_EVAL_PATH.read_text(encoding="utf-8"))["queries"]
    evaluation = evaluar_retrieval(evaluation_set, embedder=embedder)
    _write_json(DEFAULT_ARTIFACT_DIR / "retrieval_evaluation.json", evaluation)
    example = retrieve_policy_context(
        "How is debt-to-income ratio calculated for mortgage underwriting?",
        embedder=embedder,
    )
    _write_json(DEFAULT_ARTIFACT_DIR / "example_retrieval.json", example)

    run_metadata = None
    if log_mlflow:
        tracking = configurar_mlflow(
            PROJECT_ROOT,
            backend=backend,
            experiment_name=EXPERIMENT_NAME,
        )
        run_metadata = registrar_run_rag(
            tracking,
            DEFAULT_ARTIFACT_DIR,
            DEFAULT_METADATA_PATH,
            DEFAULT_EVAL_PATH,
            index_metadata,
            evaluation,
        )
        _write_json(DEFAULT_ARTIFACT_DIR / "run_metadata.json", run_metadata)
    return {
        "documents": len(documents),
        "sections": len(sections),
        "chunks": len(chunks),
        "index": index_metadata,
        "evaluation": evaluation["metrics"],
        "mlflow": run_metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Construye y evalua el RAG V2")
    parser.add_argument("--backend", choices=("local", "dagshub"), default=None)
    parser.add_argument("--skip-mlflow", action="store_true")
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()
    result = ejecutar_rag(
        backend=args.backend,
        log_mlflow=not args.skip_mlflow,
        force_download=args.force_download,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
