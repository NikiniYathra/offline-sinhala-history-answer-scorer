from __future__ import annotations

import os
from pathlib import Path
import pandas as pd
import streamlit as st

from agents import (
    AnswerScoringWorkflow,
    ExplanationAgent,
    OntologyAgent,
    QuestionAgent,
    RetrievalAgent,
    RubricAgent,
    ScoringAgent,
)
from ollama_client import OllamaClient
from ontology_engine import OntologyEngine
from rag_engine import RAGEngine
from scoring_engine import ScoringEngine


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


@st.cache_resource(show_spinner=False)
def build_workflow(model_name: str, use_ollama: bool) -> AnswerScoringWorkflow:
    question_agent = QuestionAgent(DATA_DIR / "marking_guides.json")
    rag_engine = RAGEngine(DATA_DIR / "knowledge_base")
    ontology_engine = OntologyEngine(DATA_DIR / "ontology.json")
    ollama_client = OllamaClient(model=model_name) if use_ollama else None
    scoring_engine = ScoringEngine(ollama_client=ollama_client, use_ollama=use_ollama)
    return AnswerScoringWorkflow(
        question_agent=question_agent,
        retrieval_agent=RetrievalAgent(rag_engine),
        ontology_agent=OntologyAgent(ontology_engine),
        rubric_agent=RubricAgent(),
        scoring_agent=ScoringAgent(scoring_engine),
        explanation_agent=ExplanationAgent(),
    )


def score_badge(score: int) -> str:
    if score >= 17:
        return "🌟 Excellent"
    if score >= 13:
        return "✅ Good"
    if score >= 8:
        return "⚠️ Average"
    return "🔴 Needs Improvement"


def main() -> None:
    st.set_page_config(
        page_title="Offline Sinhala History Answer Scorer",
        page_icon="📜",
        layout="wide",
    )

    st.markdown(
        """
        <style>
        .main-title {font-size: 2.2rem; font-weight: 800; margin-bottom: 0.2rem;}
        .subtitle {color: #666; margin-bottom: 1.2rem;}
        .score-card {border-radius: 18px; padding: 1.4rem; border: 1px solid rgba(120,120,120,0.2); background: rgba(240,240,240,0.35);}
        .small-muted {font-size: 0.9rem; color: #777;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="main-title">📜 Offline Sinhala Open-Ended Answer Scorer</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Anuradhapura Period History · OLLAMA + RAG + Ontology + Agent-Based Evaluation</div>',
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("⚙️ Settings")
        model_name = st.text_input("OLLAMA model", value=os.getenv("OLLAMA_MODEL", "Tharusha_Dilhara_Jayadeera/singemma"))
        use_ollama = st.toggle("Use OLLAMA scoring", value=True, help="If OLLAMA is unavailable, the app automatically uses deterministic rubric scoring.")
        top_k = st.slider("Retrieved evidence chunks", min_value=1, max_value=5, value=3)
        st.info("Run locally with: `ollama serve` and `streamlit run app.py`.")

    workflow = build_workflow(model_name, use_ollama)
    questions = workflow.question_agent.list_questions()
    labels = {f"{qid} - {question[:80]}...": qid for qid, question in questions}

    left, right = st.columns([1.05, 0.95], gap="large")

    with left:
        st.subheader("1️⃣ Select Question")
        selected_label = st.selectbox("Question", list(labels.keys()), label_visibility="collapsed")
        question_id = labels[selected_label]
        guide = workflow.question_agent.get_guide(question_id)
        st.markdown(f"**Question:** {guide['question']}")

        with st.expander("View marking guide / ලකුණු නිර්ණායක", expanded=False):
            guide_df = pd.DataFrame([
                {"Criterion": c["name"], "Marks": c["marks"], "Keywords": ", ".join(c.get("keywords", [])[:8])}
                for c in guide["criteria"]
            ])
            st.dataframe(guide_df, use_container_width=True, hide_index=True)

        st.subheader("2️⃣ Enter Student Answer")
        student_answer = st.text_area(
            "Sinhala answer",
            height=230,
            placeholder="මෙහි සිංහල පිළිතුර ඇතුළත් කරන්න...",
            label_visibility="collapsed",
        )
        evaluate = st.button("Evaluate Answer", type="primary", use_container_width=True)

    with right:
        st.subheader("System Flow")
        st.code(
            "Question → RAG Retrieval → Ontology Matching → Rubric Coverage → OLLAMA/Scoring → Explainable Output",
            language="text",
        )
        st.markdown(
            "This app runs with local files and local OLLAMA inference. The fallback scorer also works offline without internet."
        )

    if evaluate:
        if not student_answer.strip():
            st.warning("Please enter a Sinhala answer before evaluation.")
            return

        with st.spinner("Evaluating answer using RAG, ontology and scoring agents..."):
            result = workflow.evaluate(question_id, student_answer, top_k=top_k)

        score_result = result.score_result
        final_score = int(score_result.get("final_score", 0))

        st.divider()
        st.subheader("✅ Evaluation Result")
        c1, c2, c3 = st.columns([0.8, 1, 1.2])
        with c1:
            st.markdown("<div class='score-card'>", unsafe_allow_html=True)
            st.metric("Final Score", f"{final_score}/20")
            st.markdown(f"**{score_badge(final_score)}**")
            st.markdown(f"<span class='small-muted'>Mode: {score_result.get('scoring_mode', 'unknown')}</span>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with c2:
            st.markdown("**Matched Ontology Concepts**")
            concepts = score_result.get("matched_concepts") or result.ontology_analysis.get("matched_concepts", [])
            st.write(", ".join(concepts) if concepts else "No major concepts detected.")
        with c3:
            st.markdown("**Missing / Improve These Concepts**")
            missing = score_result.get("missing_concepts") or result.ontology_analysis.get("missing_keywords", [])[:10]
            st.write(", ".join(missing) if missing else "No major missing concepts detected.")

        st.markdown("### 📊 Criterion-wise Mark Breakdown")
        breakdown = score_result.get("breakdown", [])
        breakdown_df = pd.DataFrame(breakdown)
        if not breakdown_df.empty:
            st.dataframe(breakdown_df, use_container_width=True, hide_index=True)

        st.markdown("### 🧠 Overall Explanation")
        st.success(score_result.get("overall_explanation", "No explanation returned."))

        st.markdown("### ✍️ Improvement Suggestions")
        st.info(score_result.get("improvement_suggestions", "No suggestions returned."))

        st.markdown("### 🔎 Retrieved Evidence from Local Knowledge Base")
        for i, item in enumerate(result.retrieved_evidence, start=1):
            with st.expander(f"Evidence {i}: {item['source']} | similarity {item['score']}", expanded=i == 1):
                st.write(item["text"])

        st.markdown("### 🕸️ Ontology Analysis")
        with st.expander("Show ontology details", expanded=False):
            st.json(result.ontology_analysis)

        if score_result.get("ollama_error"):
            st.warning(f"OLLAMA note: {score_result['ollama_error']}")


if __name__ == "__main__":
    main()
