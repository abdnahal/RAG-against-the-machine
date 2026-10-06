from pathlib import Path
from typing import List, Dict
from .models import Chunk
from .chunking import chunk_file, save_chunks, load_chunks
from .retriever import save_bm25_index, build_postings, load_bm25_index, search, map_chunk
import hashlib
import json


def file_hash(path: Path) -> str:
    """Return a SHA-256 fingerprint of the file contents."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def discover_files(root: Path) -> list[Path]:
    files = []
    extensions = (".py", ".md", ".txt", ".rst")
    for item in root.rglob("*"):
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
) -> None:

    if not corpus_path.exists() or not corpus_path.is_dir():
        raise FileNotFoundError("Missing corpus folder!")

    processed_directory.mkdir(parents=True, exist_ok=True)
    chunks = processed_directory / 'chunks.json'
    maped = processed_directory / 'map.json'
    bm25 = processed_directory / 'bm25.json'
    if chunks.exists() and maped.exists():
        chunks_map = json.load(maped.open('r'))
        if chunks_map.get('max_chunk_size') != max_chunk_size:
            chunks_map = {}
            old_chunks = []
        else:
            old_chunks = load_chunks(chunks)
    else:
        old_chunks = []
        chunks_map = {}
    files = discover_files(corpus_path)
    files_str = [str(file) for file in files]
    old_chunks = [Chunk.model_validate(item) for item in old_chunks]
    for path in list(chunks_map.keys()):
        if path not in files_str:
            del chunks_map[path]
    chunked = collect_chunks(files, max_chunk_size, chunks_map, old_chunks)
    chunks_maped = map_chunk(chunked)
    chunk_lengths, chunk_terms, postings = build_postings(chunks_maped)
    save_chunks(chunked, chunks)
    save_bm25_index(chunk_lengths, chunk_terms, postings, bm25)
    chunks_map['max_chunk_size'] = max_chunk_size
    save_map(chunks_map, maped)


def search_saved_index(
    query: str,
    k: int,
    processed_directory: Path,
) -> list[Chunk]:
    chunks = processed_directory / 'chunks.json'
    bm25 = processed_directory / 'bm25.json'
    # if not chunks.is_file() or bm25.is_file():
    #     raise FileNotFoundError("chunks and bm25 index not found")
    try:
        chunk_lengths, chunk_terms, postings = load_bm25_index(bm25)
    except json.JSONDecodeError:
        print("Invalid json detected!\n")
        return []
    chunks_map = map_chunk(load_chunks(chunks))
    results = search(query, k, chunks_map, chunk_lengths, postings)
    return results
