from .chunking import save_chunks, load_chunks
from .indexer import discover_files, collect_chunks, save_map
from pathlib import Path
from .models import Chunk
import json
from .retriever import build_postings, map_chunk, search, save_bm25_index


if __name__ == "__main__":
    print("Rag against the machine initialized!\n")
    max_chunk_size = 2000
    # corpus = Path("data/raw/vllm-0.10.1")
    corpus = Path("src")
    posting = Path('data/processed/bm25.json')
    path = Path('data/processed/chunks.json')
    map_path = Path('data/processed/map.json')
    if path.exists() and map_path.exists():
        maped = json.load(map_path.open('r'))
        if maped['max_chunk_size'] != max_chunk_size:
            maped = {}
            chunks = []
            old_chunks = []
        else:
            chunks_loaded = json.load(path.open('r'))
            old_chunks = [Chunk.model_validate(chunk)
                          for chunk in chunks_loaded]
    else:
        maped = {}
        chunks = []
        old_chunks = []
    query = "What are the build requirements for vllm?"
    indexed = discover_files(corpus)
    strred = [str(file) for file in indexed]
    for fil in list(maped):
        if fil not in strred:
            del maped[fil]
    chunks = collect_chunks(indexed, max_chunk_size, maped, old_chunks)
    ids_map = map_chunk(chunks)
    postings = build_postings(ids_map)
    top_chunks = search(query, 5, ids_map, postings[0], postings[2])
    save_bm25_index(postings[0], postings[1], postings[2], posting)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_chunks(chunks, path)
    maped['max_chunk_size'] = max_chunk_size
    save_map(maped, Path('data/processed/map.json'))
    loaded = load_chunks(path)
