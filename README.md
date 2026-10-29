*This project has been created as part of the 42 curriculum by abdnahal.*

# RAG against the machine

## Description

RAG against the machine is a CPU-based Retrieval-Augmented Generation system for querying the supplied vLLM 0.10.1 codebase and documentation. It indexes source files, retrieves relevant passages, and uses **Qwen/Qwen3-0.6B** to generate answers from those passages.

The mandatory pipeline provides six Python Fire commands: `index`, `search`, `search_dataset`, `answer`, `answer_dataset`, and `evaluate`. Search results contain file paths and character offsets so their relevance can be checked against the reference datasets.

The implementation also includes the five bonus features: semantic embeddings, hybrid retrieval, incremental indexing, index/query caching, and a local HTTP API. Their presence does not replace validation of the mandatory requirements.

## Instructions

### Requirements and installation

- Python **3.15**: the current source uses the `lazy import` syntax and cannot run unchanged on earlier Python versions.
- [uv](https://docs.astral.sh/uv/getting-started/installation/) for Python and dependency management.
- A CPU and enough free disk space for the corpus, Python dependencies, indices, and Hugging Face model weights. Together these can occupy several gigabytes.
- Internet access for initial dependency installation and model downloads. Cached model weights can be reused afterward.

Run from the repository root:

```bash
git clone https://github.com/abdnahal/RAG-against-the-machine.git
cd RAG-against-the-machine
uv python install 3.15
uv sync --locked
```

The project configures a dedicated PyTorch CPU package index in `pyproject.toml`. Keep `uv.lock` when reproducing the environment. Python 3.15 support depends on compatible dependency wheels being available for the target platform.

Place the **provided** corpus and datasets in this layout; do not substitute another vLLM revision, because the expected source offsets refer to the supplied files:

```text
.
├── src/
├── pyproject.toml
├── uv.lock
├── Makefile
├── README.md
└── data/
    ├── raw/
    │   └── vllm-0.10.1/
    ├── datasets/
    │   ├── UnansweredQuestions/
    │   │   ├── dataset_docs_public.json
    │   │   └── dataset_code_public.json
    │   └── AnsweredQuestions/
    │       ├── dataset_docs_public.json
    │       └── dataset_code_public.json
    ├── processed/                         # Generated indices and query cache
    └── output/
        ├── search_results/<DatasetScope>/
        └── search_results_and_answer/<DatasetScope>/
```

Run commands from this directory so corpus paths match the reference datasets. Do not commit the corpus, model weights, generated indices, or generated outputs.

### Makefile commands

| Command | Purpose |
| --- | --- |
| `make install` | Install dependencies with `uv sync`. |
| `make run` | Start the Fire command interface. |
| `make debug` | Start the entry point under `pdb`. |
| `make clean` | Remove Python bytecode and mypy/pytest caches. |
| `make lint` | Run flake8 and mypy. |

`make clean` preserves saved indices, query results, and downloaded models. Use `uv run python -m src <command> ...` to pass command-specific arguments.

## Example usage

### Index and search

Build the default BM25 index:

```bash
uv run python -m src index \
    --corpus_path data/raw/vllm-0.10.1 \
    --processed_directory data/processed \
    --max_chunk_size 2000

uv run python -m src search "How do I enable LoRA adapters?" \
    --k 5 \
    --method bm25 \
    --output_path data/output/search_results/single.json
```

Search writes JSON to `--output_path`. It does not print the result JSON to the terminal. The chunk-size limit must be an integer from 1 to 2000; `k` must be a nonnegative integer.

Build or update both lexical and semantic indices before using semantic or hybrid retrieval:

```bash
uv run python -m src index --semantic=True --max_chunk_size 2000

uv run python -m src search "How does vLLM manage its key-value cache?" \
    --method hybrid --k 5 \
    --output_path data/output/search_results/hybrid_single.json
```

Use `--method semantic` to search only the vector index. Rerun `index --semantic=True` after corpus changes to update both indices together. A lexical-only update can leave an existing semantic index stale.

### Batch retrieval and evaluation

Retrieve ten sources per question for both public datasets, then measure recall at five with the local evaluator:

```bash
for scope in docs code; do
    uv run python -m src search_dataset \
        --dataset_path "data/datasets/UnansweredQuestions/dataset_${scope}_public.json" \
        --save_directory data/output/search_results/UnansweredQuestions \
        --k 10 --method bm25

    uv run python -m src evaluate \
        --student_search_results_path "data/output/search_results/UnansweredQuestions/dataset_${scope}_public.json" \
        --dataset_path "data/datasets/AnsweredQuestions/dataset_${scope}_public.json" \
        --k 5
done
```

`search_dataset` preserves question IDs and saves the input dataset's filename inside `--save_directory`. Separate output directories by dataset scope, since the answered and unanswered datasets share filenames. When comparing retrieval methods, use separate directories for each method too.

For official validation, run the supplied executable externally. The student-results path comes first and the answered ground-truth path second:

```bash
for scope in docs code; do
    ./moulinette-ubuntu evaluate_student_search_results \
        "data/output/search_results/UnansweredQuestions/dataset_${scope}_public.json" \
        "data/datasets/AnsweredQuestions/dataset_${scope}_public.json" \
        --k 10 --max_context_length 2000
done
```

Use the binary supplied for your platform, with execution permission enabled. Inspect both output validity and **Recall@5**: documentation requires at least **80%**, and code requires at least **50%**. A generic success boolean is not a substitute for checking the documentation threshold. The application does not import or invoke the moulinette.

### Generate answers

Answer a single question:

```bash
uv run python -m src answer "How do I enable LoRA adapters?" \
    --k 5 --method bm25 \
    --output data/output/search_results_and_answer/single
```

Here `--output` is a **directory**; the command writes `results.json` inside it. Answer generation downloads Qwen on its first use and runs it on the CPU.

Generate answers from the previously saved batch retrieval results:

```bash
for scope in docs code; do
    uv run python -m src answer_dataset \
        --student_search_results_path "data/output/search_results/UnansweredQuestions/dataset_${scope}_public.json" \
        --save_directory data/output/search_results_and_answer/UnansweredQuestions
done
```

`answer_dataset` reuses the saved source locations; it does not perform retrieval again. It reads the source files and slices their character ranges. Keep the corpus unchanged between retrieval and answer generation. Batch output keeps the input filename and adds an `answer` to each result.

### Local HTTP API

Build the index, then start the API from the repository root:

```bash
uv run uvicorn src.api:app --host 127.0.0.1 --port 8000
```

In another terminal:

```bash
curl http://127.0.0.1:8000/health

curl -X POST http://127.0.0.1:8000/search \
    -H 'Content-Type: application/json' \
    -d '{"query":"How do I enable LoRA adapters?","k":5,"method":"bm25"}'

curl -X POST http://127.0.0.1:8000/answer \
    -H 'Content-Type: application/json' \
    -d '{"query":"How do I enable LoRA adapters?","k":5,"method":"bm25"}'
```

Interactive API documentation is available at `http://127.0.0.1:8000/docs`. Requests require `query`, an integer `k`, and `method` (`bm25`, `semantic`, or `hybrid`). The API uses `data/processed` and returns a single result object directly, whereas CLI JSON files use the dataset wrappers.

## System architecture

```text
vLLM source files
    → file hashing and change detection
    → Python/text chunking with exact source offsets
    → persisted chunks + BM25 index [+ semantic vectors]

Question → query cache → BM25 / semantic / hybrid retrieval
                              ├→ source locations → JSON → recall evaluation
                              └→ numbered source context → Qwen → answer JSON

The CLI and local HTTP API share retrieval and generation functions.
```

| Module | Responsibility |
| --- | --- |
| `src/cli.py`, `src/__main__.py` | Fire commands, batch orchestration, command-line error reporting. |
| `src/models.py` | Pydantic schemas for chunks, questions, sources, results, and API requests. |
| `src/chunking.py` | File discovery, document segmentation, and source offsets. |
| `src/indexer.py` | Incremental indexing and retrieval-method dispatch. |
| `src/retriever.py` | Tokenization, chunk IDs, BM25 postings, and lexical ranking. |
| `src/semantic.py`, `src/hybrid.py` | Embedding storage, vector search, and rank fusion. |
| `src/cache.py` | Index revisions, in-process index reuse, and persistent query caching. |
| `src/generation.py` | Model loading, source-aware prompts, and answer generation. |
| `src/evaluation.py` | Source matching and recall calculation. |
| `src/api.py` | Health, search, and answer HTTP endpoints. |

The processed directory contains `chunks.json`, `bm25.json`, `map.json` (file hashes and chunk-size settings), and `metadata.json` (index revision). Semantic indexing adds `semantic/embeddings.npy` and `semantic/manifest.json`. Query results are cached under `cache/queries/<index_revision>/`.

Each retrieved source contains `file_path`, `first_character_index`, and `last_character_index`. Ranges are half-open: `text[start:end]`. Offsets count characters in decoded text, not bytes or tokens. Paths must match the reference dataset verbatim, and each returned span must contain at most 2000 characters.

## Chunking strategy

The indexer processes `.py`, `.md`, `.txt`, and `.rst` files with two strategies:

- **Python:** LangChain's `PythonCodeTextSplitter` uses Python-oriented separators before falling back to smaller boundaries. It preserves separators and whitespace, records start offsets, and uses zero chunk overlap. Functions or classes larger than the limit can span several chunks; this is separator-based splitting, not AST parsing.
- **Markdown and text:** a line-based builder starts a new chunk at Markdown headings or when appending a line would exceed the limit. Oversized individual lines are split into bounded pieces, and the final buffered text is emitted at end of file.

The default maximum is **2000 characters**. Chunk IDs are derived from the file path, offsets, and text, so a changed source span receives a different identity. Extra retrieval metadata, such as the embedding's file-path prefix, is not inserted into the original source text or its offsets.

Smaller chunks reduce the context carried by each result and can separate relevant information; larger chunks retain more context but may dilute lexical scores or exceed the embedding model's token window. The results below use 2000 characters. They do not establish how other chunk sizes perform; rerun evaluation after changing the limit.

## Retrieval method

### BM25

The default retriever is a custom inverted-index BM25 implementation. It lowercases text, extracts alphanumeric terms, and additionally retains full underscore-containing identifiers. The filename is appended to the searchable chunk text to help match file-specific questions.

BM25 combines term rarity, term frequency, and document-length normalization. This implementation uses `k1 = 1.2` and `b = 0.75`:

```text
IDF(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))
score(q, d) = sum over unique query terms t:
    IDF(t) × tf(t,d) × (k1 + 1)
    / (tf(t,d) + k1 × (1 - b + b × length(d) / average_length))
```

Here each chunk is a document, `N` is the number of chunks, and `df(t)` counts chunks containing a term. Results are ranked by descending score. A query with no indexed terms returns no BM25 results.

### Semantic retrieval

`sentence-transformers/all-MiniLM-L6-v2` produces normalized 384-dimensional embeddings on the CPU. Each chunk is encoded as `File: <file_path>` followed by its original text. The path supplies useful context for technical questions without changing source locations.

Queries are embedded with the same model and ranked by cosine similarity against the saved matrix. Row `i` corresponds to `chunk_ids[i]` in the semantic manifest. The model's default input window is 256 tokens, so a 2000-character chunk is not guaranteed to be encoded in full. Semantic search returns nearest neighbors without a minimum similarity threshold.

### Hybrid retrieval

Hybrid search retrieves up to `max(20, k)` candidates from each method and combines them using reciprocal rank fusion:

```text
RRF(chunk) = sum over rankings containing the chunk: 1 / (60 + rank)
```

Ranks start at one. Chunk IDs deduplicate candidates, and the top `k` fused results are returned. Rank fusion avoids directly comparing BM25 scores with cosine similarities, whose scales differ.

## Performance analysis

### Recall and validation

A source matches when its path is identical to an expected source and their character-range intersection divided by their union is at least **0.05**. Per-question recall is the fraction of expected sources matched by at least one of the top `k` results. Dataset recall is the mean of those question scores; finding the same expected source repeatedly does not increase it.

The following are **historical public-dataset measurements from development**, before the recent Python/dependency migration. They used 2000-character chunks, 100 documentation questions, 99 code questions, and retrieval of ten candidates evaluated at five. They are reference results, not a fresh certification of the current environment.

| Retrieval configuration | Docs Recall@5 | Code Recall@5 |
| --- | ---: | ---: |
| Subject minimum | 80.00% | 50.00% |
| BM25 | 83.00% | 71.72% |
| Semantic, original text only | 62.00% | 38.38% |
| Semantic, with file-path prefix | 64.00% | 44.44% |
| Hybrid, original semantic text | 81.00% | 67.68% |
| Hybrid, with file-path prefix | 84.00% | 71.72% |

Those outputs were accepted as valid by the moulinette, and local recall agreed with it. BM25 and hybrid met both recall thresholds in those runs. Semantic retrieval alone remained below both thresholds despite the improvement from adding file paths. A general-purpose embedding model and its limited input window can struggle with exact identifiers and code-specific evidence.

### Runtime and remaining limits

The subject requires full-corpus indexing within **5 minutes** and retrieval of **200 questions within 90 seconds**. Earlier CPU measurements on an Intel Core 5 120U included approximately **12 seconds** for lexical indexing and **13.5 seconds** for two BM25 batch commands covering 199 questions, including their startup time. The latter is useful context, but is not an exact 200-question timing run.

A full semantic build with file-path prefixes took approximately **557 seconds** in an earlier run, exceeding the five-minute indexing budget. Incremental updates avoid re-encoding unchanged files, but do not remove the cost of a first full build. Current timings should be measured again on the evaluation machine with the locked dependencies. Distinguish fresh builds and uncached retrieval from repeated runs served by caches.

Qwen generation is substantially slower than retrieval on the CPU. The model is reused within a batch or API process, but a separate CLI process must load it again. Generation is limited to 128 new tokens and disables Qwen's thinking mode. Answers are sampled, so wording and citation quality can vary. Recall measures retrieval only: groundedness and correct `[Source N]` citations still require inspection of generated answers.

## Design decisions

- **Lexical retrieval by default:** BM25 is fast, transparent, and effective for exact code identifiers. Embeddings and hybrid ranking remain explicit options.
- **Exact source locations:** stored chunks preserve original character ranges so retrieval, answer context, and evaluation refer to the same evidence.
- **File-level incremental updates:** SHA-256 hashes identify added, modified, and deleted files. BM25 updates the affected postings. Semantic updates encode chunks from changed files, reuse unchanged vectors by chunk ID, omit deleted chunks, and save the resulting matrix in manifest order. Changing the chunk-size setting triggers a rebuild.
- **Revision-aware caching:** changing the index changes its revision. Loaded indices are reused within a process, and disk query-cache keys include the revision, processed path, query, `k`, retrieval method, and retrieval-version tag. Generated answers are not cached. Model or embedding-text changes require rebuilding the semantic data and invalidating affected query results.
- **Source-grounded generation:** prompts label passages as `[Source N]`, include paths and offsets, request supporting citations, and ask Qwen to abstain when evidence is insufficient. These instructions reduce unsupported answers but cannot guarantee correctness.
- **Bounded generation:** Qwen runs on CPU in evaluation/inference mode. Inputs that exceed the configured 32768-token budget after reserving output space are rejected rather than silently trimmed.
- **Validated interfaces:** Pydantic models define the JSON contracts. CLI checks cover invalid `k`, duplicate IDs, incompatible evaluation cutoffs, invalid spans, and attempts to overwrite input datasets. Expected command failures are reported through the CLI error handler.

## Challenges faced

- **Losing the last chunk or overflowing on a long line:** flushing the final buffer and splitting oversized lines ensures bounded source spans.
- **Offsets diverging from source files:** preserving whitespace and slicing the original decoded text keeps retrieval and answer generation aligned with evaluation offsets.
- **Mapping vectors after incremental changes:** persisting ordered chunk IDs alongside the matrix makes the association explicit. Reused vectors must follow the new chunk order, and deleted rows must disappear.
- **Serving outdated results:** corpus revisions separate cached query results and invalidate loaded indices; semantic metadata rejects a vector index belonging to an older corpus revision.
- **Low semantic recall:** adding the file path to embedding input improved the measured results. Hybrid ranking combines that signal with stronger lexical matching, while semantic-only recall remains a limitation.
- **Slow answer generation and unsupported claims:** model reuse, bounded output, disabled thinking, and numbered evidence prompts address cost and grounding. CPU latency and small-model answer quality remain tradeoffs.
- **Python and dependency compatibility:** adopting `lazy import` raised the runtime requirement to Python 3.15. The CPU PyTorch package source and lockfile record the dependency setup; lazy imports affect startup work, not the cost of encoding or generating tokens.

## Resources

- **Project subject:** `Rag.pdf`, version 2.0, and the supplied moulinette README describe the required interfaces, validation rules, and evaluation thresholds. These are provided with the assignment.
- [Introduction to Information Retrieval — Okapi BM25](https://nlp.stanford.edu/IR-book/html/htmledition/okapi-bm25-a-non-binary-model-1.html): probabilistic ranking, term frequency, and length normalization.
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401): the retrieval-plus-generation approach.
- [Qwen3-0.6B model card](https://huggingface.co/Qwen/Qwen3-0.6B): model usage, chat templates, and generation settings.
- [all-MiniLM-L6-v2 model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2): sentence embeddings and model limitations.
- [Python Fire guide](https://google.github.io/python-fire/guide/): exposing Python methods as CLI commands.
- [Pydantic documentation](https://docs.pydantic.dev/latest/): JSON schemas and validation.
- [uv: Using PyTorch](https://docs.astral.sh/uv/guides/integration/pytorch/): dependency installation and CPU package sources.
- [FastAPI documentation](https://fastapi.tiangolo.com/): request models and HTTP endpoints.

### AI assistance

**This README was created with the help of AI (OpenAI Codex/ChatGPT).** AI helped read the subject, inspect the implementation, organize the required sections, and draft the installation, architecture, usage, and evaluation explanations.

During development, AI assistance was also used to explain BM25, source offsets, caching, embeddings, and incremental indexing; review code and diagnose errors; suggest targeted fixes and helper functions; and assist with test cases, retrieval benchmarks, and prompt experiments. Reported benchmark results are separate from claims about current correctness: they should be reproduced after implementation or environment changes.
