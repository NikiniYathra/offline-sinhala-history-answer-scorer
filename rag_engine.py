"""Offline TF-IDF RAG engine for Sinhala history answer scoring."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class RetrievedChunk:
    source: str
    text: str
    score: float


class RAGEngine:
    """Loads local markdown knowledge files and retrieves relevant evidence chunks offline."""

    def __init__(self, knowledge_dir: str | Path = "data/knowledge_base", chunk_size: int = 850, overlap: int = 120):
        self.knowledge_dir = Path(knowledge_dir)
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.chunks: List[RetrievedChunk] = []
        self.vectorizer: TfidfVectorizer | None = None
        self.matrix = None
        self._load_and_index()

    def _read_markdown_files(self) -> list[tuple[str, str]]:
        if not self.knowledge_dir.exists():
            raise FileNotFoundError(f"Knowledge base folder not found: {self.knowledge_dir}")
        files = sorted(self.knowledge_dir.glob("*.md"))
        if not files:
            raise FileNotFoundError(f"No .md files found inside {self.knowledge_dir}")
        docs: list[tuple[str, str]] = []
        for file in files:
            text = file.read_text(encoding="utf-8")
            text = re.sub(r"\s+", " ", text).strip()
            docs.append((file.name, text))
        return docs

    def _split_text(self, text: str) -> list[str]:
        if len(text) <= self.chunk_size:
            return [text]
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start = max(end - self.overlap, start + 1)
        return chunks

    def _load_and_index(self) -> None:
        docs = self._read_markdown_files()
        raw_chunks: list[RetrievedChunk] = []
        for source, text in docs:
            for chunk in self._split_text(text):
                raw_chunks.append(RetrievedChunk(source=source, text=chunk, score=0.0))

        self.chunks = raw_chunks
        corpus = [chunk.text for chunk in self.chunks]
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=1)
        self.matrix = self.vectorizer.fit_transform(corpus)

    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievedChunk]:
        if not query.strip():
            return []
        if self.vectorizer is None or self.matrix is None:
            self._load_and_index()
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.matrix).flatten()
        ranked_indices = similarities.argsort()[::-1][:top_k]
        results: list[RetrievedChunk] = []
        for idx in ranked_indices:
            chunk = self.chunks[int(idx)]
            results.append(RetrievedChunk(source=chunk.source, text=chunk.text, score=float(similarities[idx])))
        return results

    @staticmethod
    def format_evidence(chunks: list[RetrievedChunk]) -> str:
        if not chunks:
            return "No retrieved evidence found."
        lines = []
        for i, chunk in enumerate(chunks, start=1):
            lines.append(f"Evidence {i} | Source: {chunk.source} | Similarity: {chunk.score:.3f}\n{chunk.text}")
        return "\n\n".join(lines)
