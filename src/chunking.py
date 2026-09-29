from .models import Chunk
from langchain_text_splitters import PythonCodeTextSplitter


def chunk_markdown(
    content: str,
    file_path: str,
    max_chunk_size: int = 2000,
) -> list[Chunk]:
    """Split Markdown into chunks."""

    chunks: list[Chunk] = []

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
) -> list[Chunk]:
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
