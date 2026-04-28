"""Scoring engine combining deterministic rubric coverage with optional OLLAMA reasoning."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import json
import re

from ollama_client import OllamaClient


@dataclass
class CriterionScore:
    criterion: str
    max_marks: int
    awarded_marks: int
    reason: str
    matched_keywords: list[str]
    missing_keywords: list[str]


def normalize_text(text: str) -> str:
    text = text or ""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def keyword_present(text: str, keyword: str) -> bool:
    return normalize_text(keyword) in normalize_text(text)


class ScoringEngine:
    """Creates explainable marks out of 20. OLLAMA is used when available; deterministic fallback is always available."""

    def __init__(self, ollama_client: OllamaClient | None = None, use_ollama: bool = True):
        self.ollama_client = ollama_client
        self.use_ollama = use_ollama

    def deterministic_score(self, answer: str, guide: dict[str, Any], ontology_analysis: dict[str, Any]) -> dict[str, Any]:
        answer_norm = normalize_text(answer)
        answer_word_count = len(answer_norm.split())
        breakdown: list[CriterionScore] = []

        for criterion in guide["criteria"]:
            keywords = criterion.get("keywords", [])
            max_marks = int(criterion["marks"])
            matched = [kw for kw in keywords if keyword_present(answer, kw)]
            missing = [kw for kw in keywords if kw not in matched]

            if not keywords:
                raw = 0
            else:
                coverage = len(matched) / len(keywords)
                raw = round(coverage * max_marks)

            # Reward clear, sufficiently developed answers; penalize extremely short answers.
            if answer_word_count < 8:
                awarded = min(raw, max(1 if matched else 0, max_marks // 3))
            elif answer_word_count < 25:
                awarded = min(raw, max_marks - 1) if raw == max_marks and max_marks > 1 else raw
            else:
                awarded = raw

            awarded = max(0, min(max_marks, int(awarded)))
            reason = self._build_sinhala_reason(awarded, max_marks, matched, missing)
            breakdown.append(
                CriterionScore(
                    criterion=criterion["name"],
                    max_marks=max_marks,
                    awarded_marks=awarded,
                    reason=reason,
                    matched_keywords=matched,
                    missing_keywords=missing[:6],
                )
            )

        total = sum(item.awarded_marks for item in breakdown)
        total = max(0, min(20, total))
        return {
            "final_score": total,
            "breakdown": [item.__dict__ for item in breakdown],
            "matched_concepts": ontology_analysis.get("matched_concepts", []),
            "missing_concepts": ontology_analysis.get("missing_keywords", [])[:12],
            "overall_explanation": self._overall_explanation(total),
            "improvement_suggestions": self._improvement_suggestions(guide, breakdown),
            "scoring_mode": "deterministic_rubric_fallback"
        }

    @staticmethod
    def _build_sinhala_reason(awarded: int, max_marks: int, matched: list[str], missing: list[str]) -> str:
        if awarded == max_marks:
            return f"මෙම කොටස ඉතා හොඳින් ආවරණය කර ඇත. සඳහන් කළ කරුණු: {', '.join(matched[:6])}."
        if awarded == 0:
            return f"මෙම නිර්ණායකයට අදාළ ප්‍රධාන කරුණු ප්‍රමාණවත් ලෙස සඳහන් කර නැත. අවශ්‍ය කරුණු: {', '.join(missing[:6])}."
        return f"මෙම කොටස අර්ධ වශයෙන් ආවරණය කර ඇත. සඳහන් කරුණු: {', '.join(matched[:5])}. වැඩිදියුණු කළ යුතු කරුණු: {', '.join(missing[:5])}."

    @staticmethod
    def _overall_explanation(score: int) -> str:
        if score >= 17:
            return "පිළිතුර ඉතා ශක්තිමත්ය. ප්‍රධාන ඉතිහාස කරුණු, සම්බන්ධතා සහ පැහැදිලි කිරීම හොඳින් ආවරණය කර ඇත."
        if score >= 13:
            return "පිළිතුර සාමාන්‍යයෙන් හොඳය. ප්‍රධාන කරුණු කිහිපයක් නිවැරදිව දක්වා ඇති නමුත් තවදුරටත් උදාහරණ සහ ගැඹුරු පැහැදිලි කිරීම අවශ්‍ය වේ."
        if score >= 8:
            return "පිළිතුර මධ්‍යම මට්ටමේය. මූලික කරුණු කිහිපයක් ඇතත්, නිර්ණායක බොහොමයක් සම්පූර්ණයෙන් ආවරණය වී නැත."
        return "පිළිතුර දුර්වලය. ප්‍රධාන ඉතිහාස කරුණු, උදාහරණ සහ සම්බන්ධතා වැඩිදුරටත් එක් කළ යුතුය."

    @staticmethod
    def _improvement_suggestions(guide: dict[str, Any], breakdown: list[CriterionScore]) -> str:
        weak = [b for b in breakdown if b.awarded_marks < b.max_marks]
        if not weak:
            return "තවදුරටත් වැඩිදියුණු කිරීමට, කරුණු කාල අනුපිළිවෙළට සහ හේතු-ප්‍රතිඵල සම්බන්ධතාවය සමඟ ඉදිරිපත් කරන්න."
        suggestions = []
        for item in weak[:3]:
            missing = ", ".join(item.missing_keywords[:4]) or "අදාළ උදාහරණ"
            suggestions.append(f"'{item.criterion}' සඳහා {missing} වැනි කරුණු එකතු කරන්න")
        return "; ".join(suggestions) + "."

    def build_prompt(self, question: str, answer: str, guide: dict[str, Any], evidence: str, ontology_text: str, deterministic: dict[str, Any]) -> str:
        return f"""
You are an offline Sinhala History answer evaluator for the Anuradhapura Period.
Evaluate the student's Sinhala answer out of 20 using ONLY the marking guide, retrieved evidence, and ontology analysis.
Do not invent new historical claims. Give a fair criterion-wise score.

Question:
{question}

Student Answer:
{answer}

Marking Guide JSON:
{json.dumps(guide, ensure_ascii=False, indent=2)}

Retrieved Evidence:
{evidence}

Ontology Analysis:
{ontology_text}

A deterministic rubric pre-score is provided below. You may adjust it slightly only if the answer meaning clearly deserves it, but every criterion must stay within its max marks and total must be <= 20.
{json.dumps(deterministic, ensure_ascii=False, indent=2)}

Return ONLY valid JSON in this format:
{{
  "final_score": 0,
  "breakdown": [
    {{
      "criterion": "",
      "max_marks": 0,
      "awarded_marks": 0,
      "reason": "Sinhala explanation"
    }}
  ],
  "matched_concepts": [],
  "missing_concepts": [],
  "overall_explanation": "Sinhala explanation",
  "improvement_suggestions": "Sinhala suggestions",
  "scoring_mode": "ollama_rag_ontology"
}}
""".strip()

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any] | None:
        text = text.strip()
        try:
            return json.loads(text)
        except Exception:
            pass
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except Exception:
            return None

    @staticmethod
    def validate_result(result: dict[str, Any], guide: dict[str, Any]) -> dict[str, Any]:
        guide_criteria = guide.get("criteria", [])
        breakdown = result.get("breakdown", [])
        fixed = []
        for i, criterion in enumerate(guide_criteria):
            item = breakdown[i] if i < len(breakdown) and isinstance(breakdown[i], dict) else {}
            max_marks = int(criterion["marks"])
            awarded = int(item.get("awarded_marks", 0) or 0)
            awarded = max(0, min(max_marks, awarded))
            fixed.append({
                "criterion": criterion["name"],
                "max_marks": max_marks,
                "awarded_marks": awarded,
                "reason": item.get("reason") or "මෙම නිර්ණායකයට අනුව ලකුණු ලබා දී ඇත."
            })
        result["breakdown"] = fixed
        result["final_score"] = max(0, min(20, sum(x["awarded_marks"] for x in fixed)))
        result.setdefault("matched_concepts", [])
        result.setdefault("missing_concepts", [])
        result.setdefault("overall_explanation", "පිළිතුර ලකුණු නිර්ණායක අනුව ඇගයීමට ලක් කරන ලදී.")
        result.setdefault("improvement_suggestions", "අඩු වූ නිර්ණායක සඳහා තවදුරටත් නිශ්චිත ඉතිහාස කරුණු හා උදාහරණ එක් කරන්න.")
        result.setdefault("scoring_mode", "validated")
        return result

    def score(self, question: str, answer: str, guide: dict[str, Any], evidence: str, ontology_analysis: dict[str, Any], ontology_text: str) -> dict[str, Any]:
        deterministic = self.deterministic_score(answer, guide, ontology_analysis)
        if not self.use_ollama or self.ollama_client is None:
            return self.validate_result(deterministic, guide)

        prompt = self.build_prompt(question, answer, guide, evidence, ontology_text, deterministic)
        response = self.ollama_client.generate(prompt)
        if not response.success:
            deterministic["ollama_error"] = response.error
            return self.validate_result(deterministic, guide)

        parsed = self._extract_json(response.text)
        if parsed is None:
            deterministic["ollama_raw_response"] = response.text[:1200]
            deterministic["ollama_error"] = "OLLAMA returned non-JSON output, so deterministic fallback was used."
            return self.validate_result(deterministic, guide)
        return self.validate_result(parsed, guide)
