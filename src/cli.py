from pathlib import Path
from .indexer import index_corpus, search_saved_index, map_chunk
from .indexer import load_bm25_index
from tqdm import tqdm
from .generation import generate_answer, build_messages, load_llm
from .retriever import search as ret_search
from .chunking import load_chunks
from .evaluation import recall_for_question
from .models import MinimalSource, MinimalSearchResults, UnansweredQuestion
from .models import StudentSearchResults, RagDataset, AnsweredQuestion
from .models import StudentSearchResultsAndAnswer, MinimalAnswer, Chunk
import json


class RagCLI:
    """Expose the RAG pipeline through command-line commands."""

    def index(
        self,
        max_chunk_size: int = 2000,
        corpus_path: str = "data/raw/vllm-0.10.1",
        processed_directory: str = "data/processed",
    ) -> None:
        """Build or update the saved index."""
        if max_chunk_size > 2000 or max_chunk_size <= 0:
            raise ValueError("Invalid max_chunk_size!")
        corpus = Path(corpus_path)
        processed = Path(processed_directory)
        index_corpus(corpus, processed, max_chunk_size)
        print("Indexing completed successfully!")

    def search(
        self,
        query: str,
        k: int = 5,
        processed_directory: str = "data/processed",
        output_path: str = "data/output/search_results/single.json",
    ) -> None:
        """Search the saved index and write JSON results."""
        sources = []
        if not isinstance(k, int) or k < 0:
            raise ValueError("Top k results should be postive!")
        if query.strip() and k:
            chunks = search_saved_index(query, k, Path(processed_directory))
            for chunk in chunks:
                sources.append(
                    MinimalSource(
                        file_path=chunk.file_path,
                        first_character_index=chunk.first_character_index,
                        last_character_index=chunk.last_character_index,
                    )
                )
        question = UnansweredQuestion(question=query)
        results = MinimalSearchResults(
            question_id=question.question_id,
            question=question.question,
            retrieved_sources=sources,
        )
        student_results = StudentSearchResults(search_results=[results], k=k)
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w") as f:
            json.dump(student_results.model_dump(), f, indent=2)

    def search_dataset(
        self,
        dataset_path: str,
        k: int,
        save_directory: str,
        processed_directory: str = "data/processed",
    ) -> None:
        """Search every question in a dataset."""
        results = []
        if not isinstance(k, int) or k < 0:
            raise ValueError("Top k results should be postive!")
        with open(dataset_path, "r") as f:
            data_loaded = json.load(f)
            dataset = RagDataset.model_validate(data_loaded)
        chunks = load_chunks(Path(processed_directory) / "chunks.json")
        chunk_ids = map_chunk(chunks)
        chunk_lengths, _, postings = load_bm25_index(
            Path(processed_directory) / "bm25.json"
        )
        for question in tqdm(dataset.rag_questions, desc="Searching"):
            ret_chunks = ret_search(
                question.question, k, chunk_ids, chunk_lengths, postings
            )
            sources = [
                MinimalSource(
                    file_path=chunk.file_path,
                    first_character_index=chunk.first_character_index,
                    last_character_index=chunk.last_character_index,
                )
                for chunk in ret_chunks
            ]
            results.append(
                MinimalSearchResults(
                    question=question.question,
                    question_id=question.question_id,
                    retrieved_sources=sources,
                )
            )
        student_results = StudentSearchResults(
            search_results=results,
            k=k,
        )

        path = Path(save_directory) / Path(dataset_path).name
        path.parent.mkdir(parents=True, exist_ok=True)

        path.write_text(
            student_results.model_dump_json(indent=2),
            encoding="utf-8",
        )

    def evaluate(
        self, student_search_results_path: str, dataset_path: str, k: int = 5
    ) -> None:
        """Report retrieval recall against ground-truth sources."""
        if type(k) is not int or k < 0:
            raise ValueError("k must be a nonnegative integer.")

        student_search = StudentSearchResults.model_validate_json(
            Path(student_search_results_path).read_text(encoding="utf-8")
        )
        dataset = RagDataset.model_validate_json(
            Path(dataset_path).read_text(encoding="utf-8")
        )

        if k > student_search.k:
            raise ValueError("Requested cutoff exceeds the saved search k.")

        if not dataset.rag_questions:
            raise ValueError("The ground-truth dataset is empty.")

        expected_ids = set()
        for question in dataset.rag_questions:
            if not isinstance(question, AnsweredQuestion):
                raise ValueError("Dataset should contain answered questions")
            if question.question_id in expected_ids:
                raise KeyError("Duplicated question id detected!")
            for src in question.sources:
                if not (
                    0 <= src.first_character_index < src.last_character_index
                ):
                    raise ValueError("Invalid ground-truth offsets!")
            if not question.sources:
                raise ValueError(
                    f"Question {question.question_id} has no expected sources."
                )
            expected_ids.add(question.question_id)

        result_ids = {}

        for result in student_search.search_results:
            if result.question_id in result_ids.keys():
                raise ValueError("Duplicated question id detected!")
            for src in result.retrieved_sources:
                if not (
                    0 <= src.first_character_index < src.last_character_index
                ):
                    raise ValueError("Invalid retrieved-source offsets!")
                length = src.last_character_index - src.first_character_index
                if length > 2000:
                    raise ValueError("Large chunk detected (max = 2000) !")
            result_ids[result.question_id] = result.retrieved_sources

        unknown_ids = set(result_ids.keys()) - expected_ids
        if unknown_ids:
            raise ValueError(
                f"Results contain unknown question IDs: {sorted(unknown_ids)}"
            )

        recalls = []
        for question in dataset.rag_questions:
            ret = result_ids.get(question.question_id, [])
            recalls.append(recall_for_question(ret, question.sources, k))
        avg = sum(recalls) / len(recalls)
        print(f"Recall@{k}: {avg:.3f}")

    def answer(self, query: str,
               output: str = 'data/output/search_results',
               processed_directory: str = 'data/processed',
               k: int = 5) -> None:
        if type(k) is not int or k < 0:
            raise ValueError("k must be a nonnegative integer.")

        tokenizer, model = load_llm()
        chunks = []
        path = Path(output) / "results.json"
        if query.strip() and k > 0:
            chunks = search_saved_index(
                query,
                k,
                Path(processed_directory),
            )

        if not query.strip():
            answer_text = "Please provide a nonempty question."
        elif not chunks:
            answer_text = "No relevant sources were found to answer this \
question."
        else:
            messages = build_messages(query, chunks)
            answer_text = generate_answer(tokenizer, model, messages)
        sources = [
            MinimalSource(
                file_path=c.file_path,
                first_character_index=c.first_character_index,
                last_character_index=c.last_character_index,
            )
            for c in chunks
        ]
        q = UnansweredQuestion(question=query)
        result = MinimalAnswer(
            question_id=q.question_id,
            question=q.question,
            retrieved_sources=sources,
            answer=answer_text,
        )
        output = StudentSearchResultsAndAnswer(search_results=[result],
                                               k=k)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w') as f:
            json.dump(output.model_dump(), f, indent=2)

    def answer_dataset(
        self,
        student_search_results_path: str,
        save_directory: str = "data/output/search_results_and_answer",
    ) -> None:
        answers = []
        tokenizer, model = load_llm()
        path = Path(student_search_results_path)
        results = StudentSearchResults.model_validate_json(
                            path.read_text(encoding="utf-8")
                        )
        file_cache = {}
        for result in tqdm(results.search_results, desc="Answering"):
            chunks = []
            for file in result.retrieved_sources:
                if file.file_path not in file_cache.keys():
                    with open(file.file_path, 'r') as f:
                        content = f.read()
                    file_cache[file.file_path] = content
                else:
                    content = file_cache[file.file_path]
                start = file.first_character_index
                end = file.last_character_index
                if not (0 <= start < end <= len(content)):
                    raise ValueError("Invalid source offsets.")

                if end - start > 2000:
                    raise ValueError("Source exceeds 2000 characters.")
                obj = Chunk(text=content[start:end],
                            file_path=file.file_path,
                            first_character_index=start,
                            last_character_index=end)
                chunks.append(obj)
            if not result.question.strip():
                re = "Please provide a nonempty question."
            elif not chunks:
                re = "No relevant sources were found."
            else:
                messages = build_messages(result.question, chunks)
                re = generate_answer(tokenizer, model, messages)
            answers.append(MinimalAnswer(question=result.question,
                           question_id=result.question_id,
                           retrieved_sources=result.retrieved_sources,
                           answer=re))
        output = StudentSearchResultsAndAnswer(search_results=answers,
                                               k=results.k)
        output_path = Path(save_directory) / path.name
        output_path.parent.mkdir(parents=True, exist_ok=True)

        output_path.write_text(
            output.model_dump_json(indent=2),
            encoding="utf-8",
        )
