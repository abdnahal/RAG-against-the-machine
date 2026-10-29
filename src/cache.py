from pathlib import Path
lazy import json
import uuid
from .models import IndexData, Chunk
from functools import lru_cache
from .chunking import load_chunks
from .retriever import load_bm25_index, map_chunk
from hashlib import sha256
from pydantic import ValidationError


def write_index_revision(processed_directory: Path) -> str:
    metadata = Path(processed_directory) / "metadata.json"
    uid = str(uuid.uuid4())
    with metadata.open('w') as f:
        json.dump({"index_revision": uid}, f)
    return uid


def read_index_revision(processed_directory: Path) -> str:
    metadata = Path(processed_directory) / 'metadata.json'
    with metadata.open('r') as f:
        try:
            data = json.load(f)
            if type(data) is not dict:
                raise TypeError("Invalid index type!")
            if type(data['index_revision']) is not str:
                raise TypeError("Invalid index_revision type !")
            if not data['index_revision'].strip():
                raise ValueError("Empty index revision uuid detected!")
        except json.JSONDecodeError:
            raise ValueError("Invalid Json in metadata.json!")
        except KeyError:
            raise KeyError("Invalid key, expected 'index_revision' !")
    return data['index_revision']


def load_index_cached(processed_directory: Path) -> IndexData:
    path = Path(processed_directory).resolve()
    data = path / 'metadata.json'
    chunks = path / 'chunks.json'
    bm25 = path / 'bm25.json'

    valid = all(
        file.is_file()
        for file in (data, chunks, bm25)
    )
    if valid:
        revision = read_index_revision(path)
        index = _load_index_for_revision(str(path), revision)
    else:
        raise FileNotFoundError("Missing index files, try rebuilding them !")
    return index


@lru_cache(maxsize=1)
def _load_index_for_revision(
    directory: str,
    revision: str,
) -> IndexData:
    bm25 = Path(directory) / 'bm25.json'
    chunks_f = Path(directory) / 'chunks.json'
    chunks = load_chunks(chunks_f)
    chunk_lengths, _, postings = load_bm25_index(bm25)
    maped = map_chunk(chunks)
    index = IndexData(chunks_ids=maped,
                      chunk_lengths=chunk_lengths,
                      postings=postings)
    return index


def write_query_cache(
    cache_path: Path,
    chunks: list[Chunk],
) -> None:
    cache_path.parent.mkdir(exist_ok=True, parents=True)
    with cache_path.open('w') as f:
        objects = [chunk.model_dump() for chunk in chunks]
        json.dump(objects, f)


def read_query_cache(cache_path: Path) -> list[Chunk] | None:
    if not cache_path.exists() or not cache_path.is_file():
        return None
    with cache_path.open('r') as f:
        try:
            parsed = json.load(f)
            if type(parsed) is not list:
                return None
            chunks = [Chunk.model_validate(chunk) for chunk in parsed]
        except (json.JSONDecodeError, ValidationError):
            return None
    return chunks


def make_query_cache_key(
    processed_directory: Path,
    revision: str,
    query: str,
    k: int,
    method: str = 'bm25'
) -> str:
    payload = {
        "retrieval_version": "v2",
        "directory": str(Path(processed_directory).resolve()),
        "index_revision": revision,
        "query": query,
        "k": k,
        "method": method.lower(),
    }

    content = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
    )
    return sha256(content.encode("utf-8")).hexdigest()
