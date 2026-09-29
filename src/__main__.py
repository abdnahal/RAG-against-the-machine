from .chunking import chunk_code

if __name__ == "__main__":
    print("Rag against the machine initialized!\n")
    with open("src/chunking.py", 'r') as f:
        content = f.read()
        chunks = chunk_code(content, "src/chunking.py", 2000)
