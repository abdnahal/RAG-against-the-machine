from .models import Chunk
from langchain_text_splitters import PythonCodeTextSplitter
from typing import List
from pathlib import Path
import json


def chunk_markdown(
    content: str,
    file_path: str,
    max_chunk_size: int = 2000,
) -> List[Chunk]:
    """Split Markdown into chunks."""

    chunks: List[Chunk] = []

    lines = content.splitlines(keepends=True)

    current_text = ""
    chunk_start = 0
    position = 0
    if max_chunk_size <= 0:
        raise ValueError("max_chunk_size must be greater than 0")

    for line in lines:
        if len(line) > max_chunk_size:
            if current_text:
                chunks.append(Chunk(
                    text=current_text,
                    file_path=file_path,
                    first_character_index=chunk_start,
                    last_character_index=position,
                ))
                current_text = ""

            start = 0
            while start < len(line):
                piece = line[start:start + max_chunk_size]

                chunks.append(Chunk(
                    text=piece,
                    file_path=file_path,
                    first_character_index=position,
                    last_character_index=position + len(piece),
                ))

                position += len(piece)
                start += len(piece)

            chunk_start = position
            continue
        if current_text and len(current_text) + len(line) > max_chunk_size:
            chunks.append(Chunk(text=current_text,
                                file_path=file_path,
                                first_character_index=chunk_start,
                                last_character_index=position))
            current_text = line
            chunk_start = position
            position += len(current_text)
            continue
        if line.startswith('#') and current_text:
            chunks.append(Chunk(text=current_text,
                                file_path=file_path,
                                first_character_index=chunk_start,
                                last_character_index=position))
            current_text = line
            chunk_start = position
            position += len(current_text)
        else:
            current_text += line
            position += len(line)

    if current_text:
        chunks.append(Chunk(text=current_text,
                            file_path=file_path,
                            first_character_index=chunk_start,
                            last_character_index=position))

    return chunks


def chunk_code(
    content: str,
    file_path: str,
    max_chunk_size: int = 2000,
) -> List[Chunk]:
    """Spit code into chunks"""
    splitter = PythonCodeTextSplitter(chunk_size=max_chunk_size,
                                      chunk_overlap=0,
                                      keep_separator=True,
                                      strip_whitespace=False,
                                      add_start_index=True,)
    document = splitter.create_documents([content])
    chunks = []
    for doc in document:
        text = doc.page_content
        start = doc.metadata["start_index"]
        end = start + len(text)
        chunks.append(Chunk(text=text,
                            file_path=file_path,
                            first_character_index=start,
                            last_character_index=end))
    return chunks


def save_chunks(chunks: List[Chunk], path: Path) -> None:
    final = []
    with open(str(path), 'w') as f:
        for chunk in chunks:
            final.append(chunk.model_dump())
        json.dump(final, f, indent=2)


def chunk_file(path: Path, max_chunk_size: int) -> list[Chunk]:
    chunks = []
    path_str = str(path)
    with open(path_str, 'r') as f:
        if path_str.endswith('.py'):
            chunks = chunk_code(f.read(), path_str, max_chunk_size)
        else:
            chunks = chunk_markdown(f.read(), path_str, max_chunk_size)
    return chunks


def load_chunks(path: Path) -> List[Chunk]:
    chunks = []
    with open(str(path), 'r', encoding='utf-8') as f:
        loaded = json.load(f)
        for dct in loaded:
            chunks.append(Chunk.model_validate(dct))
    return chunks
