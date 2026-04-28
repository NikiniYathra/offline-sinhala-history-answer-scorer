"""Agent-based workflow for the Sinhala history answer scorer."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

from rag_engine import RAGEngine
from ontology_engine import OntologyEngine
from scoring_engine import ScoringEngine


@dataclass
class EvaluationResult:
    question_id: str
    question: str
    student_answer: str
    retrieved_evidence: list[dict[str, Any]]
    ontology_analysis: dict[str, Any]
    score_result: dict[str, Any]


class QuestionAgent:
    """Loads questions and marking guides."""

    def __init__(self, marking_guide_path: str | Path = "data/marking_guides.json"):
        self.marking_guide_path = Path(marking_guide_path)
        self.guides = self._load_guides()

    def _load_guides(self) -> dict[str, Any]:
        if not self.marking_guide_path.exists():
            raise FileNotFoundError(f"Marking guide file not found: {self.marking_guide_path}")
        with self.marking_guide_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def list_questions(self) -> list[tuple[str, str]]:
        return [(qid, data["question"]) for qid, data in self.guides.items()]

    def get_guide(self, question_id: str) -> dict[str, Any]:
        if question_id not in self.guides:
            raise KeyError(f"Unknown question id: {question_id}")
        return self.guides[question_id]


class RetrievalAgent:
    """Retrieves local evidence using the RAG engine."""

    def __init__(self, rag_engine: RAGEngine):
        self.rag_engine = rag_engine

    def run(self, question: str, answer: str, top_k: int = 3):
        query = f"{question}\n{answer}"
        return self.rag_engine.retrieve(query, top_k=top_k)


class OntologyAgent:
    """Analyzes student answer against ontology and expected rubric terms."""

    def __init__(self, ontology_engine: OntologyEngine):
        self.ontology_engine = ontology_engine

    @staticmethod
    def expected_keywords_from_guide(guide: dict[str, Any]) -> list[str]:
        keywords = []
        for criterion in guide.get("criteria", []):
            keywords.extend(criterion.get("keywords", []))
        # preserve order, remove duplicates
        seen = set()
        ordered = []
        for keyword in keywords:
            if keyword not in seen:
                ordered.append(keyword)
                seen.add(keyword)
        return ordered

    def run(self, answer: str, guide: dict[str, Any]) -> dict[str, Any]:
        expected = self.expected_keywords_from_guide(guide)
        return self.ontology_engine.analyze(answer, expected_keywords=expected)


class RubricAgent:
    """Provides rubric coverage information before final scoring."""

    @staticmethod
    def run(answer: str, guide: dict[str, Any]) -> dict[str, Any]:
        answer_lower = answer.lower()
        coverage = []
        for criterion in guide.get("criteria", []):
            keywords = criterion.get("keywords", [])
            matched = [kw for kw in keywords if kw.lower() in answer_lower]
            coverage.append({
                "criterion": criterion["name"],
                "max_marks": criterion["marks"],
                "matched_keywords": matched,
                "coverage_ratio": round(len(matched) / len(keywords), 3) if keywords else 0,
            })
        return {"criteria_coverage": coverage}


class ScoringAgent:
    """Runs the final scoring engine."""

    def __init__(self, scoring_engine: ScoringEngine):
        self.scoring_engine = scoring_engine

    def run(self, question: str, answer: str, guide: dict[str, Any], evidence_text: str, ontology_analysis: dict[str, Any], ontology_text: str) -> dict[str, Any]:
        return self.scoring_engine.score(question, answer, guide, evidence_text, ontology_analysis, ontology_text)


class ExplanationAgent:
    """Formats output for the UI/report."""

    @staticmethod
    def evidence_to_dicts(chunks) -> list[dict[str, Any]]:
        return [
            {"source": chunk.source, "score": round(chunk.score, 4), "text": chunk.text}
            for chunk in chunks
        ]


class AnswerScoringWorkflow:
    """Complete agent workflow: question → retrieval → ontology → rubric → scoring → explanation."""

    def __init__(self, question_agent: QuestionAgent, retrieval_agent: RetrievalAgent, ontology_agent: OntologyAgent, rubric_agent: RubricAgent, scoring_agent: ScoringAgent, explanation_agent: ExplanationAgent):
        self.question_agent = question_agent
        self.retrieval_agent = retrieval_agent
        self.ontology_agent = ontology_agent
        self.rubric_agent = rubric_agent
        self.scoring_agent = scoring_agent
        self.explanation_agent = explanation_agent

    def evaluate(self, question_id: str, student_answer: str, top_k: int = 3) -> EvaluationResult:
        guide = self.question_agent.get_guide(question_id)
        question = guide["question"]
        retrieved_chunks = self.retrieval_agent.run(question, student_answer, top_k=top_k)
        evidence_text = self.retrieval_agent.rag_engine.format_evidence(retrieved_chunks)
        ontology_analysis = self.ontology_agent.run(student_answer, guide)
        ontology_text = self.ontology_agent.ontology_engine.format_analysis(ontology_analysis)
        rubric_coverage = self.rubric_agent.run(student_answer, guide)
        ontology_analysis["rubric_coverage"] = rubric_coverage
        score_result = self.scoring_agent.run(question, student_answer, guide, evidence_text, ontology_analysis, ontology_text)
        return EvaluationResult(
            question_id=question_id,
            question=question,
            student_answer=student_answer,
            retrieved_evidence=self.explanation_agent.evidence_to_dicts(retrieved_chunks),
            ontology_analysis=ontology_analysis,
            score_result=score_result,
        )
