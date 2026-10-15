from pathlib import Path
import json
import uuid


def write_index_revision(processed_directory: Path) -> str:
    metadata = Path(processed_directory) / "metadata.json"
    uid = str(uuid.uuid4())
    with metadata.open('w') as f:
        json.dump({"index_revision": uid}, f)
    return uid


def read_index_revision(processed_directory: Path) -> str:
    metadata = Path(processed_directory) / 'metadata.json'
    with metadata.open('r') as f:
        data = json.load(f)
    return data.get('index_revision', "")
