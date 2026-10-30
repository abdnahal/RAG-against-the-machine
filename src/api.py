from fastapi import FastAPI, HTTPException
from pathlib import Path
from .models import (MinimalSource, MinimalSearchResults,
                     UnansweredQuestion, MinimalAnswer,
                     QueryRequest)
from .indexer import search_saved_index
from .generation import load_llm, build_messages, generate_answer

app = FastAPI()


@app.get('/health')
def health():
    return {'status': 'ok'}


def search(query: str,
           k: int = 5,
           processed_directory: str = "data/processed",
           method: str = "bm25") -> MinimalSearchResults:
    sources = []
    if type(k) is not int or k < 0:
        raise HTTPException(status_code=422,
                            detail="k must be a nonnegative integer.")
    if query.strip() and k:
        chunks = search_saved_index(query, k,
                                    Path(processed_directory),
                                    method=method)
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
    return results


def answer(query: str,
           processed_directory: str = 'data/processed',
           k: int = 5,
           method: str = "bm25") -> MinimalAnswer:

    if type(k) is not int or k < 0:
        raise HTTPException(status_code=422,
                            detail="k must be a nonnegative integer.")

    chunks = []
    if query.strip() and k > 0:
        chunks = search_saved_index(
            query,
            k,
            Path(processed_directory),
            method=method
        )

    if not query.strip():
        answer_text = "Please provide a nonempty question."
    elif not chunks:
        answer_text = "No relevant sources were found to answer this \
question."
    else:
        tokenizer, model = load_llm()
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
    return result


@app.post('/search', response_model=MinimalSearchResults)
def search_endpoint(request: QueryRequest) -> MinimalSearchResults:
    try:
        search_result = search(request.query, request.k, method=request.method)
    except FileNotFoundError:
        raise HTTPException(status_code=503,
                            detail="Required local files are missing. Check \
the index and model files.")
    return search_result


@app.post('/answer', response_model=MinimalAnswer)
def answer_question(request: QueryRequest) -> MinimalAnswer:
    try:
        answer_result = answer(request.query,
                               k=request.k,
                               method=request.method)
    except FileNotFoundError:
        raise HTTPException(status_code=503,
                            detail="Required local files are missing. Check \
the index and model files.")
    return answer_result
