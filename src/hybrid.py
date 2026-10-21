from .models import Chunk
from .retriever import map_chunk
from collections import defaultdict


def fuse_rankings(
    bm25_chunks: list[Chunk],
    semantic_chunks: list[Chunk],
    k: int,
    rank_constant: int = 60,
) -> list[Chunk]:
    if rank_constant < 0:
        raise ValueError("Rank constant should be 0 or greater!")
    chunk_ids = map_chunk(semantic_chunks + bm25_chunks)
    bm25_ids = map_chunk(bm25_chunks)
    semantic_ids = map_chunk(semantic_chunks)
    scores = defaultdict(float)
    for rank, id in enumerate(bm25_ids, start=1):
        scores[id] += 1 / (rank_constant + rank)
    for rank, id in enumerate(semantic_ids, start=1):
        scores[id] += 1 / (rank_constant + rank)
    result = sorted(scores, key=lambda id: scores[id], reverse=True)
    return [chunk_ids[idd] for idd in result[:k]]
