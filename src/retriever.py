import re
from collections import Counter
from typing import List, Dict, Tuple
import json
from .models import Chunk
import hashlib
import math
from pathlib import Path


def tokenize(text: str) -> list[str]:
    """Extract lowercase terms and preserve complete snake_case identifiers."""
    text = text.lower()

    tokens = re.findall(r"[a-z0-9]+", text)
    identifiers = re.findall(r"\b[a-z_][a-z0-9_]*\b", text)

    tokens.extend(
        identifier
        for identifier in identifiers
        if "_" in identifier
        and any(character.isalnum() for character in identifier)
    )
    return tokens


def counted(sequence: List) -> Dict[str, int]:
    """Return the frequency of each item in a sequence."""
    count = Counter(sequence)
    return count


def map_chunk(chunks: List[Chunk]) -> Dict[str, Chunk]:
    """Map stable chunk identifiers to their source chunks."""
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
    """Build token lengths, per-chunk counts, and inverted postings."""
    chunk_lengths = {}
    chunk_terms = {}
    postings: dict[str, dict[str, int]] = {}
    for chunk_id, chunk in maped.items():
        searchable_text = (
            chunk.text + "\n" + Path(chunk.file_path).name
            )
        tokens = tokenize(searchable_text)
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
    """Return one query term's BM25 contribution to a chunk score."""
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
    """Return up to k chunks ranked by BM25 score."""
    if k <= 0 or not chunk_lengths:
        return []
    chunks = []
    scores: dict[str, float] = {}
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
    """Save chunk lengths, term counts, and postings as JSON."""
    data = {
        'chunk_lengths': chunk_lengths,
        'chunk_terms': chunk_terms,
        'postings': postings,
    }
    with open(str(path), 'w') as f:
        json.dump(data, f, indent=2)


def load_bm25_index(path: Path) -> Tuple[
    Dict[str, int],
    Dict[str, Dict[str, int]],
    Dict[str, Dict[str, int]],
]:
    """Load chunk lengths, term counts, and postings from JSON."""
    with open(str(path), 'r') as f:
        data = json.load(f)
        return (data['chunk_lengths'], data["chunk_terms"], data["postings"])
