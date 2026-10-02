from pathlib import Path
from typing import List, Dict
from .models import Chunk
from .chunking import chunk_file
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
