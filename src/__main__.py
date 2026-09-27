from .chunking import chunk_markdown, chunk_code

if __name__ == "__main__":
    print("Rag against the machine initialized!\n")
    with open("src/models.py", 'r') as f:
        content = f.read()
        chunk_code(content, None, None)
