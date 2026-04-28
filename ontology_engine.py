"""Ontology concept matching for the Anuradhapura-period answer scorer."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json


@dataclass
class OntologyMatch:
    concept: str
    concept_type: str
    matched_terms: list[str]
    related_concepts: list[str]
    relationships: list[dict[str, str]]


class OntologyEngine:
    """Loads a lightweight JSON ontology and detects concepts in Sinhala answers."""

    def __init__(self, ontology_path: str | Path = "data/ontology.json"):
        self.ontology_path = Path(ontology_path)
        self.ontology: dict[str, Any] = self._load_ontology()

    def _load_ontology(self) -> dict[str, Any]:
        if not self.ontology_path.exists():
            raise FileNotFoundError(f"Ontology file not found: {self.ontology_path}")
        with self.ontology_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _contains_term(text: str, term: str) -> bool:
        return term.strip().lower() in text.lower()

    def analyze(self, answer: str, expected_keywords: list[str] | None = None) -> dict[str, Any]:
        answer = answer or ""
        matches: list[OntologyMatch] = []

        for concept, data in self.ontology.items():
            terms = data.get("sinhala_terms", []) + [concept]
            matched_terms = [term for term in terms if self._contains_term(answer, term)]
            if matched_terms:
                matches.append(
                    OntologyMatch(
                        concept=concept,
                        concept_type=data.get("type", "Concept"),
                        matched_terms=matched_terms,
                        related_concepts=data.get("related_concepts", []),
                        relationships=data.get("relationships", []),
                    )
                )

        expected_keywords = expected_keywords or []
        missing_keywords = [kw for kw in expected_keywords if not self._contains_term(answer, kw)]
        matched_keywords = [kw for kw in expected_keywords if self._contains_term(answer, kw)]

        return {
            "matched_concepts": [m.concept for m in matches],
            "matched_terms": sorted({term for m in matches for term in m.matched_terms}),
            "related_concepts": sorted({rel for m in matches for rel in m.related_concepts}),
            "relationships": [rel for m in matches for rel in m.relationships],
            "matched_keywords": matched_keywords,
            "missing_keywords": missing_keywords,
            "details": [m.__dict__ for m in matches],
        }

    @staticmethod
    def format_analysis(analysis: dict[str, Any]) -> str:
        matched = ", ".join(analysis.get("matched_concepts", [])) or "None"
        terms = ", ".join(analysis.get("matched_terms", [])) or "None"
        missing = ", ".join(analysis.get("missing_keywords", [])) or "None"
        relations = analysis.get("relationships", [])
        relation_text = "; ".join([f"{r.get('relation')} → {r.get('target')}" for r in relations]) or "None"
        return (
            f"Matched ontology concepts: {matched}\n"
            f"Matched Sinhala terms: {terms}\n"
            f"Missing expected keywords: {missing}\n"
            f"Relevant ontology relationships: {relation_text}"
        )
