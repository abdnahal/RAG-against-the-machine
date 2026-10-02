import re
from collections import Counter
from typing import List, Dict, Tuple
import json
from .models import Chunk
import hashlib
import math
from pathlib import Path


def tokenize(text: str) -> list[str]:
    text = text.lower()
    sequence = re.findall('[a-z0-9]+', text)
    return sequence


def counted(sequence: List) -> Dict[str, int]:
    count = Counter(sequence)
    return count


def map_chunk(chunks: List[Chunk]) -> Dict[str, Chunk]:
    maped = {}
    for chunk in chunks:
        maped[make_chunk_id(chunk)] = chunk
    return maped


def make_chunk_id(chunk: Chunk) -> str:
    """Create a stable identifier for a source chunk."""
    identity = [
        chunk.file_path,
        chunk.first_character_index,
        chunk.last_character_index,
        chunk.text,
    ]
    encoded = json.dumps(identity, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_postings(maped: Dict[str, Chunk]) -> Tuple[
    Dict[str, int],
    Dict[str, Dict[str, int]],
    Dict[str, Dict[str, int]],
]:
    chunk_lengths = {}
    chunk_terms = {}
    postings = {}
    for chunk_id, chunk in maped.items():
        tokens = tokenize(chunk.text)
        if not tokens:
            continue
        count = counted(tokens)
        chunk_lengths[chunk_id] = len(tokens)
        chunk_terms[chunk_id] = count
        for term, frequency in count.items():
            postings.setdefault(term, {})[chunk_id] = frequency
    return (chunk_lengths, chunk_terms, postings)


def bm25_term_score(
    tf: int,
    df: int,
    document_count: int,
    document_length: int,
    average_length: float,
    k1: float = 1.2,
    b: float = 0.75,
) -> float:
    idf = math.log(
        1 + (document_count - df + 0.5) / (df + 0.5)
    )

    length_adjustment = (
        1 - b + b * document_length / average_length
    )

    frequency_weight = (
        tf * (k1 + 1)
        / (tf + k1 * length_adjustment)
    )

    return idf * frequency_weight


def search(
    query: str,
    k: int,
    chunks_by_id: dict[str, Chunk],
    chunk_lengths: dict[str, int],
    postings: dict[str, dict[str, int]],
) -> list[Chunk]:
    if k <= 0 or not chunk_lengths:
        return []
    chunks = []
    scores = {}
    N = sum(chunk_lengths.values())
    avg = N / len(chunk_lengths)
    for term in sorted(set(tokenize(query))):
        term_chunks = postings.get(term, {})
        df = len(term_chunks)
        for id, tf in term_chunks.items():
            scores[id] = scores.get(id, 0) + bm25_term_score(
                tf, df, len(chunk_lengths), chunk_lengths[id], avg, 1.2,  0.75)
    top = sorted(scores, key=lambda x: scores[x], reverse=True)
    chunks.extend([chunks_by_id[id] for id in top[:k]])
    return chunks


def save_bm25_index(
    chunk_lengths: dict[str, int],
    chunk_terms: dict[str, dict[str, int]],
    postings: dict[str, dict[str, int]],
    path: Path,
) -> None:
    data = {}
    with open(str(path), 'w') as f:
        data['chunk_lengths'] = chunk_lengths
        data['chunk_terms'] = chunk_terms
        data['postings'] = postings
        json.dump(data, f, indent=2)
