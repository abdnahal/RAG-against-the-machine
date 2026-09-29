from .chunking import chunk_code
from .indexer import discover_files
from pathlib import Path


if __name__ == "__main__":
    print("Rag against the machine initialized!\n")
    corpus = Path("vllm-0.10.1")
    indexed = discover_files(corpus)
    print(len(indexed))
    # for file in indexed:
    #     print(str(file))
    # indexed_str = str(indexed[1])
    # with open(indexed_str, 'r') as f:
    #     chunks = chunk_code(f.read(), indexed_str, 2000)
    #     for chunk in chunks:
    #         print(chunk.file_path, chunk.text, chunk.first_character_index, chunk.last_character_index)
