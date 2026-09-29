from pathlib import Path


def discover_files(root: Path) -> list[Path]:
    files = []
    other = []
    extensions = (".py", ".md", ".txt", ".rst")
    for item in root.rglob("*"):
        if item.is_file():
            if item.name.endswith(extensions):
                files.append(item)
            else:
                other.append(item)
    for f in other:
        files.append(f)
    return files
