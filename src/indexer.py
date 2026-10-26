from .chunking import chunk_file, save_chunks, load_chunks
from .retriever import save_bm25_index, build_postings, load_bm25_index
from .retriever import search, map_chunk
from .cache import (
    read_index_revision, write_index_revision,
    load_index_cached, make_query_cache_key,
    read_query_cache, write_query_cache)
from .hybrid import fuse_rankings
from .semantic import search_semantic, read_semantic_manifest
from pathlib import Path
from typing import List, Dict
from .models import Chunk
from tqdm import tqdm
import hashlib
import json
import os
from typing import Any


def file_hash(path: Path) -> str:
    """Return a SHA-256 fingerprint of the file contents."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def discover_files(root: Path) -> list[Path]:
    files = []
    extensions = (".py", ".md", ".txt", ".rst")
    for item in tqdm(root.rglob("*"), desc="Indexing"):
        if item.is_file():
            if item.name.endswith(extensions):
                files.append(item)
    return files


def collect_chunks(files: List[Path],
                   max_chunk_size: int,
                   indexed: Dict[str, str],
                   old_chunks: List[Chunk]) -> List[Chunk]:
    chunks = []
    for file in files:
        file_str = str(file)
        hashed = file_hash(file)
        if hashed == indexed.get(file_str):
            chunks.extend([chunk for chunk in old_chunks
                           if chunk.file_path == file_str])
            continue
        else:
            chunks.extend(chunk_file(file, max_chunk_size))
            indexed[file_str] = hashed
    return chunks


def save_map(maped: Dict[str, str], path: Path) -> None:
    with open(str(path), 'w') as f:
        json.dump(maped, f, indent=2)


def index_corpus(
    corpus_path: Path,
    processed_directory: Path,
    max_chunk_size: int = 2000
) -> dict[str, Any]:

    if not corpus_path.exists() or not corpus_path.is_dir():
        raise FileNotFoundError("Missing corpus folder!")

    corpus_path = Path(
        os.path.relpath(corpus_path.resolve(), Path.cwd())
    )
    processed_directory.mkdir(parents=True, exist_ok=True)
    chunks = processed_directory / 'chunks.json'
    maped = processed_directory / 'map.json'
    bm25 = processed_directory / 'bm25.json'
    index_metadata = processed_directory / 'metadata.json'

    rebuild = not all(
        path.is_file()
        for path in (chunks, maped, bm25, index_metadata)
    )

    if chunks.exists() and maped.exists() and index_metadata.exists():
        with maped.open('r') as f:
            chunks_map = json.load(f)
            if chunks_map.get('max_chunk_size') != max_chunk_size:
                rebuild = True
                chunks_map = {}
                old_chunks = []
            else:
                old_chunks = load_chunks(chunks)
        read_index_revision(processed_directory)
    else:
        old_chunks = []
        chunks_map = {}
    files = discover_files(corpus_path)
    files_str = [str(file) for file in files]
    old_chunks = [Chunk.model_validate(item) for item in old_chunks]
    old_hashes = {file: hashed for file, hashed in chunks_map.items()
                  if file != 'max_chunk_size'}
    old_chunk_ids = map_chunk(old_chunks)
    new_hashes = {str(file): file_hash(file) for file in files}
    added = new_hashes.keys() - old_hashes.keys()
    deleted = old_hashes.keys() - new_hashes.keys()
    modified = {file for file in new_hashes.keys() & old_hashes.keys()
                if new_hashes[file] != old_hashes[file]}
    for path in list(chunks_map.keys()):
        if path not in files_str:
            del chunks_map[path]
    chunked = collect_chunks(files, max_chunk_size, chunks_map, old_chunks)
    chunks_maped = map_chunk(chunked)
    ids = {id for id, chunk in old_chunk_ids.items()
           if chunk.file_path in modified | deleted}
    new_chunks = {id: chunk for id, chunk in chunks_maped.items()
                  if chunk.file_path in added | modified}
    if not rebuild:
        if not (added | deleted | modified):
            return {
                "added": added,
                "modified": modified,
                "deleted": deleted,
                "rebuild": rebuild
            }

        chunk_lengths, chunk_terms, postings = load_bm25_index(bm25)

        for id in ids:
            old_terms = chunk_terms.pop(id, {})

            for term in old_terms:
                postings[term].pop(id, None)

                if not postings[term]:
                    del postings[term]
            chunk_lengths.pop(id, None)

        new_lengths, new_terms, new_postings = build_postings(new_chunks)

        chunk_lengths.update(new_lengths)
        chunk_terms.update(new_terms)

        for term, occurrences in new_postings.items():
            postings.setdefault(term, {}).update(occurrences)
    else:
        chunk_lengths, chunk_terms, postings = build_postings(chunks_maped)

    save_chunks(chunked, chunks)
    save_bm25_index(chunk_lengths, chunk_terms, postings, bm25)
    chunks_map['max_chunk_size'] = max_chunk_size
    save_map(chunks_map, maped)
    write_index_revision(processed_directory)
    return {
            "added": added,
            "modified": modified,
            "deleted": deleted,
            "rebuild": rebuild
            }


def search_saved_index(
    query: str,
    k: int,
    processed_directory: Path,
    method: str = 'bm25'
) -> list[Chunk]:
    if type(query) is not str or type(k) is not int or k < 0:
        raise ValueError("Invalid parameter detected!")
    if type(method) is not str or method.lower() not in ('bm25', 'semantic',
                                                         'hybrid'):
        raise ValueError("Method should be a string (bm25 or semantic)")
    if k == 0 or not query.strip():
        return []
    method = method.lower()
    path = processed_directory.resolve()
    metadata = path / 'metadata.json'
    chunkss = path / 'chunks.json'
    revision = read_index_revision(path)
    hashed = make_query_cache_key(path, revision, query, k, method)
    cache_path = path / "cache" / "queries" / revision / f"{hashed}.json"
    required_files = [metadata, chunkss]

    if method in ("bm25", "hybrid"):
        required_files.append(path / "bm25.json")

    if method in ("semantic", "hybrid"):
        required_files.extend([
            path / "semantic" / "manifest.json",
            path / "semantic" / "embeddings.npy",
        ])

    for file in required_files:
        if not file.is_file():
            raise FileNotFoundError(f"Missing index file: {file}")

    if method in ("semantic", "hybrid"):
        read_semantic_manifest(path)
    chunks = read_query_cache(cache_path)
    if chunks is not None:
        return chunks

    if method in ('bm25', 'hybrid'):
        index = load_index_cached(path)
        retrieval_k = max(20, k) if method == "hybrid" else k
        bm25_results = search(query, retrieval_k,
                              index.chunks_ids,
                              index.chunk_lengths,
                              index.postings)
        if method != "hybrid":
            results = bm25_results
    if method in ("semantic", 'hybrid'):
        if cache_path.is_file():
            chunks = read_query_cache(cache_path)
            if chunks is not None:
                return chunks
        retrieval_k = max(20, k) if method == "hybrid" else k
        semantic_results = search_semantic(query, path, retrieval_k)
        if method != "hybrid":
            results = semantic_results
    if method == "hybrid":
        results = fuse_rankings(bm25_results, semantic_results, k)
    write_query_cache(cache_path, results)
    return results
