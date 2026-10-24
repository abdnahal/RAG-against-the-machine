# from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from functools import lru_cache
from .chunking import load_chunks
from .retriever import map_chunk
from pathlib import Path
import numpy as np
from .cache import read_index_revision
import json
from .models import Chunk
from torch import topk
from typing import Any


@lru_cache(maxsize=1)
def load_embedding_model() -> SentenceTransformer:
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2',
                                device='cpu')
    return model


def read_semantic_manifest(processed_directory: Path) -> dict:
    """Read semantic metadata and reject an incompatible or stale index."""
    directory = Path(processed_directory)
    manifest_path = directory / "semantic" / "manifest.json"

    if not manifest_path.is_file():
        raise FileNotFoundError("Build the semantic index first!")

    try:
        with manifest_path.open("r", encoding="utf-8") as file:
            manifest = json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError("Invalid JSON in semantic manifest.") from exc

    if not isinstance(manifest, dict):
        raise ValueError("Semantic manifest must be a dictionary.")

    required = {"index_revision", "model_name", "dimension", "chunk_ids"}
    missing = required - manifest.keys()
    if missing:
        raise ValueError(f"Missing manifest fields: {sorted(missing)}")

    revision = manifest["index_revision"]
    if not isinstance(revision, str) or not revision.strip():
        raise ValueError("Invalid semantic index revision.")

    if revision != read_index_revision(directory):
        raise ValueError("Semantic index is outdated. Rebuild it.")

    expected_model = "sentence-transformers/all-MiniLM-L6-v2"
    if manifest["model_name"] != expected_model:
        raise ValueError("Semantic model mismatch. Rebuild the index.")

    dimension = manifest["dimension"]
    if type(dimension) is not int or dimension <= 0:
        raise ValueError("Embedding dimension must be a positive integer.")

    chunk_ids = manifest["chunk_ids"]
    if not isinstance(chunk_ids, list):
        raise ValueError("chunk_ids must be a list.")
    if any(not isinstance(chunk_id, str) for chunk_id in chunk_ids):
        raise ValueError("Every chunk ID must be a string.")
    if len(set(chunk_ids)) != len(chunk_ids):
        raise ValueError("Duplicate chunk IDs in semantic manifest.")

    return manifest


@lru_cache(maxsize=1)
def _load_semantic_for_revision(
    directory: str,
    revision: str,
    model_name: str
            ) -> tuple[list[str], np.ndarray, dict[str, Chunk]]:
    """Load and remember semantic data for one index revision."""
    path = Path(directory)
    manifest = read_semantic_manifest(path)

    if (
        manifest["index_revision"] != revision
        or manifest["model_name"] != model_name
    ):
        raise ValueError("Semantic index changed while loading. Retry.")

    embeddings = np.load(
        path / "semantic" / "embeddings.npy",
        allow_pickle=False,
    )
    chunk_ids = manifest["chunk_ids"]

    if (
        embeddings.ndim != 2
        or embeddings.shape[1] != manifest["dimension"]
    ):
        raise ValueError("Incompatible embedding dimension.")

    if embeddings.shape[0] != len(chunk_ids):
        raise ValueError("Chunk IDs and embedding rows do not match.")

    chunks_by_id = map_chunk(load_chunks(path / "chunks.json"))

    for chunk_id in chunk_ids:
        if chunk_id not in chunks_by_id:
            raise ValueError(f"Unknown chunk ID: {chunk_id}")

    return chunk_ids, embeddings, chunks_by_id


def load_semantic_index(
    processed_directory: Path,
) -> tuple[list[str], np.ndarray, dict[str, Chunk]]:
    """Validate current metadata, then reuse loaded semantic data."""
    path = Path(processed_directory).resolve()
    manifest = read_semantic_manifest(path)

    required_files = (
        path / "chunks.json",
        path / "semantic" / "embeddings.npy",
    )
    for file in required_files:
        if not file.is_file():
            raise FileNotFoundError(f"Missing index file: {file}")

    return _load_semantic_for_revision(
        str(path),
        manifest["index_revision"],
        manifest["model_name"]
    )


def build_semantic_index(processed_directory: Path) -> None:
    path = processed_directory / 'chunks.json'
    embeddings = processed_directory / 'semantic' / 'embeddings.npy'
    data = processed_directory / 'semantic' / 'manifest.json'
    embeddings.parent.mkdir(parents=True, exist_ok=True)
    chunks = load_chunks(path)
    maped = map_chunk(chunks)
    ids = list(maped.keys())
    texts = [f"File: {maped[id].file_path}\n\n" + maped[id].text
             for id in ids]
    manifest = {'index_revision': read_index_revision(processed_directory)}
    if not texts:
        manifest['dimension'] = 384
        chunks_embeddings = np.ndarray((0, 384), dtype=np.float32)
    else:
        model = load_embedding_model()
        chunks_embeddings = model.encode_document(texts,
                                                  normalize_embeddings=True,
                                                  convert_to_numpy=True,
                                                  batch_size=16,
                                                  show_progress_bar=True)
        manifest['dimension'] = chunks_embeddings.shape[1]
    manifest['model_name'] = "sentence-transformers/all-MiniLM-L6-v2"
    manifest['chunk_ids'] = ids
    np.save(embeddings, chunks_embeddings)
    with data.open('w') as f:
        json.dump(manifest, f)


def search_semantic(
    query: str,
    processed_directory: Path,
    k: int = 5
) -> list[Chunk]:

    if type(k) is not int or k < 0 or type(query) is not str:
        raise ValueError("k must be a nonnegative integer.")

    if k == 0 or not query.strip():
        return []

    chunk_ids, embeddings, maped = load_semantic_index(processed_directory)
    if not chunk_ids:
        return []
    model = load_embedding_model()
    encoded = model.encode_query(query, normalize_embeddings=True,
                                 convert_to_numpy=True)
    top_k = min(k, len(chunk_ids))
    similarity = model.similarity(embeddings, encoded).squeeze(1)
    scores, indices = topk(similarity, k=top_k)
    result = [maped[chunk_ids[idx]] for idx in indices.tolist()]
    return result
