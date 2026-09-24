"""Pydantic models used by the RAG pipeline."""

import uuid

from pydantic import BaseModel, Field


class MinimalSource(BaseModel):
    """Represent a source location in the corpus."""

    file_path: str
    first_character_index: int
    last_character_index: int


class Chunk(BaseModel):
    """Represent one searchable piece of a source file."""

    text: str
    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    """Represent a question without an answer."""

    question_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())
    )
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """Represent a question with ground-truth sources and answer."""

    sources: list[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """Represent a dataset of RAG questions."""

    rag_questions: list[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    """Represent retrieved sources for one question."""

    question_id: str
    question: str
    retrieved_sources: list[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """Represent retrieval results together with an answer."""

    answer: str


class StudentSearchResults(BaseModel):
    """Represent retrieval results for multiple questions."""

    search_results: list[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """Represent generated answers for multiple questions."""

    search_results: list[MinimalAnswer]
    k: int
