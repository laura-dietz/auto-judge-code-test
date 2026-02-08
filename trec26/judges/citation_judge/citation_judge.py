#!/usr/bin/env python3
"""
CitationJudge: Citation validation using Auto-ARGUE framework.

Evaluates citation quality by:
1. Validating citations exist in documents dictionary
2. Using Auto-ARGUE attestation prompt to verify citations support claims
3. Tracking citation accuracy and support metrics
"""
from typing import Sequence, Optional, Literal
from pydantic import BaseModel
import dspy
from textwrap import dedent
import re
import asyncio
from pathlib import Path
import sys

# Add parent directory to path for debug_logger import
sys.path.insert(0, str(Path(__file__).parent.parent))
from debug_logger import DebugLogger

from trec_auto_judge import (
    AutoJudge,
    Report,
    Request,
    Leaderboard,
    LeaderboardBuilder,
    LeaderboardSpec,
    MeasureSpec,
    Qrels,
    QrelsSpec,
    MinimaLlmConfig,
    OpenAIMinimaLlm,
    build_qrels,
    doc_id_md5,
)
from trec_auto_judge.leaderboard.leaderboard import mean_of_floats, mean_of_bools
from trec_auto_judge.nugget_data import NuggetBanksProtocol
from trec_auto_judge.llm.minima_llm_dspy import run_dspy_batch


class CitationAssessment(BaseModel):
    """Assessment of a single citation."""
    run_id: str
    topic_id: str
    sentence: str
    citation_id: str
    document: str  # Changed from document_text to match AttestationPrompt

    # LLM outputs (filled after assessment)
    citation_exists: Optional[bool] = None  # Does citation exist in documents?
    citation_supports: Optional[bool] = None  # Does cited doc support sentence?


def _parse_yes_no(s: str) -> bool:
    """Extract YES/NO from Auto-ARGUE response."""
    s_upper = s.upper().strip()
    if 'YES' in s_upper:
        return True
    elif 'NO' in s_upper:
        return False
    else:
        # Default to NO if unclear
        return False


def _convert_attestation_output(prediction: dspy.Prediction, alignment: BaseModel) -> None:
    """Convert Auto-ARGUE YES/NO output to boolean."""
    alignment.citation_supports = _parse_yes_no(prediction.answer)


class AttestationPrompt(dspy.Signature):
    """Auto-ARGUE attestation prompt for checking if sentence is supported by document."""

    __doc__ = dedent("""
        You are an expert at determining if statements are supported by a document.
        Your task is to determine if a sentence's claims are supported by a provided document.
        A sentence is supported by a document if and only if is entailed by the document.
        Respond with ONLY 'YES' or 'NO' in English.

        Is the following sentence supported by the provided document?

        Examples:
        1. Sentence: Deep learning models have achieved 98% accuracy on the ImageNet dataset.
        Document: Deep learning models have revolutionized computer vision tasks, consistently outperforming traditional methods on large-scale image classification benchmarks. Notably, state-of-the-art architectures such as convolutional neural networks (CNNs) and transformer-based models have pushed accuracy levels to 98% on the ImageNet dataset.
        Answer (YES/NO): YES

        2. Sentence: The sky appears blue due to Rayleigh scattering.
        Document: The sky exhibits different colors depending on the time of day and atmospheric conditions. During sunrise and sunset, it often takes on shades of red and orange as light passes through a thicker layer of the atmosphere.
        Answer (YES/NO): NO

        3. Sentence: Recent studies have shown a 15% increase in global temperatures.
        Document: Recent studies have revealed significant changes in global climate patterns, including a 15% increase in global temperatures. Scientists attribute this rise to factors such as greenhouse gas emissions and deforestation.
        Answer (YES/NO): YES

        Sentence: {sentence}
        Document: {document}
        Answer (YES/NO):
    """)

    sentence: str = dspy.InputField(desc="The sentence to check")
    document: str = dspy.InputField(desc="The document that should support the sentence")

    answer: Literal["YES", "NO"] = dspy.OutputField(
        desc="YES if sentence is supported by document, NO otherwise"
    )

    convert_output = staticmethod(_convert_attestation_output)


# Qrels spec for citation assessments
CITATION_QRELS = QrelsSpec(
    topic_id=lambda x: x.topic_id,
    doc_id=lambda x: doc_id_md5(f"{x.sentence}_{x.citation_id}"),
    grade=lambda x: 1.0 if (x.citation_exists and x.citation_supports) else 0.0,
)

# Leaderboard spec for citation metrics
CITATION_SPEC = LeaderboardSpec(measures=(
    MeasureSpec("CITATION_ACCURACY", aggregate=mean_of_floats, cast=float, default=0.0),
    MeasureSpec("CITATION_SUPPORT", aggregate=mean_of_floats, cast=float, default=0.0),
    MeasureSpec("AVG_CITATIONS", aggregate=mean_of_floats, cast=float, default=0.0),
    MeasureSpec("PERFECT_CITATIONS", aggregate=mean_of_bools, cast=bool, default=False),
))


class CitationJudge(AutoJudge):
    """Judge that validates citations using Auto-ARGUE framework."""

    def __init__(self, settings: Optional[dict] = None, debug_log: Optional[str] = None, **kwargs):
        super().__init__(settings=settings, **kwargs)
        # Topic format: auto-detect or explicit
        self.topic_format = settings.get("topic_format", "auto") if settings else "auto"
        # Debug logger
        self.debug_logger = DebugLogger(debug_log)

    def extract_query(self, topic: Request, topic_format: str = "auto") -> str:
        """Extract query from topic based on dataset format."""
        # Same logic as DirectPromptJudge for consistency
        if topic_format == "auto":
            # Auto-detect
            if hasattr(topic, 'problem_statement') and topic.problem_statement:
                topic_format = "ragtime"
            elif hasattr(topic, 'body') and topic.body:
                topic_format = "dragun"
            else:
                topic_format = "rag"

        if topic_format == "ragtime":
            # RAGTIME: Combine title + problem_statement + background
            parts = [topic.title]
            if hasattr(topic, 'problem_statement') and topic.problem_statement:
                parts.append(topic.problem_statement)
            if hasattr(topic, 'background') and topic.background:
                parts.append(topic.background)
            return " ".join(parts)

        elif topic_format == "dragun":
            # DRAGUN: Use body (document text)
            return topic.body if hasattr(topic, 'body') else topic.title

        else:  # rag
            # RAG: Use title only
            return topic.title

    def split_into_sentences(self, text: str) -> list[str]:
        """Split text into sentences (simple version)."""
        # Simple sentence splitting on periods, exclamation, question marks
        # This is a basic implementation - could be improved with spacy or nltk
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]

    def create_qrels(
        self,
        rag_responses: Sequence[Report],
        rag_topics: Sequence[Request],
        llm_config: MinimaLlmConfig,
        nugget_banks: Optional[NuggetBanksProtocol] = None,
        **kwargs
    ) -> Qrels:
        """Assess citations for each response."""

        # Get topic_format from kwargs
        topic_format = kwargs.get("topic_format", self.topic_format)

        # Build topic lookup
        topic_dict = {req.request_id: req for req in rag_topics}

        # Prepare citation assessments
        assessments = []

        print(f"Analyzing citations in {len(rag_responses)} responses...")

        for response in rag_responses:
            topic = topic_dict.get(response.metadata.topic_id)
            if not topic:
                continue

            # Get documents dictionary
            documents = response.documents or {}

            # Parse responses array
            responses_array = response.responses or []

            # Track citation stats for this response
            total_citations = 0

            for resp_segment in responses_array:
                sentence = resp_segment.text if hasattr(resp_segment, 'text') else ''
                citations_raw = resp_segment.citations if hasattr(resp_segment, 'citations') else []

                # Handle both dict (RAGTIME) and list (RAG/DRAGUN) citations
                if isinstance(citations_raw, dict):
                    # RAGTIME: citations is Dict[str, float]
                    citations = list(citations_raw.keys())
                elif citations_raw is None:
                    citations = []
                else:
                    # RAG/DRAGUN: citations is List[str] or List[int]
                    citations = [str(c) for c in citations_raw]

                # Count citations
                total_citations += len(citations)

                # Assess each citation
                for citation_id in citations:
                    # Check if citation exists
                    citation_exists = citation_id in documents

                    if citation_exists:
                        # Document is a Document object with .text attribute
                        doc = documents[citation_id]
                        document = doc.get_text() if hasattr(doc, 'get_text') else doc.text
                    else:
                        document = ""

                    assessments.append(CitationAssessment(
                        run_id=response.metadata.run_id,
                        topic_id=response.metadata.topic_id,
                        sentence=sentence,
                        citation_id=citation_id,
                        document=document,
                        citation_exists=citation_exists,
                    ))

        print(f"Assessing {len(assessments)} citations with Auto-ARGUE attestation prompt...")

        # Log inputs if debug enabled
        if self.debug_logger.enabled:
            for assessment in assessments:
                log_data = {
                    "sentence": assessment.sentence,
                    "citation_id": assessment.citation_id,
                    "citation_exists": assessment.citation_exists,
                    "document_length": len(assessment.document),
                }

                # Include full prompt for existing citations (similar to direct_prompt_judge)
                if assessment.citation_exists:
                    # Build complete prompt as it would be sent to LLM
                    template = AttestationPrompt.__doc__
                    complete_prompt = template.format(
                        sentence=assessment.sentence,
                        document=assessment.document
                    )
                    log_data["prompt_template"] = complete_prompt
                    log_data["document_text"] = assessment.document

                self.debug_logger.log(
                    f"CITATION_INPUT [{assessment.run_id} / {assessment.topic_id}]",
                    log_data
                )

        # Run attestation checks via DSPy (only for existing citations)
        citations_to_check = [a for a in assessments if a.citation_exists]

        if citations_to_check:
            checked = asyncio.run(run_dspy_batch(
                AttestationPrompt,
                citations_to_check,
                AttestationPrompt.convert_output,
                backend=OpenAIMinimaLlm(llm_config)
            ))

            # Update assessments with results
            checked_lookup = {id(a): a for a in checked}
            for assessment in assessments:
                if id(assessment) in checked_lookup:
                    assessment.citation_supports = checked_lookup[id(assessment)].citation_supports
                else:
                    # Citation doesn't exist, so it can't support the sentence
                    assessment.citation_supports = False
        else:
            # No citations to check
            for assessment in assessments:
                assessment.citation_supports = False

        # Log outputs if debug enabled
        for assessment in assessments:
            self.debug_logger.log(
                f"CITATION_OUTPUT [{assessment.run_id} / {assessment.topic_id}]",
                {
                    "citation_id": assessment.citation_id,
                    "citation_exists": assessment.citation_exists,
                    "citation_supports": assessment.citation_supports,
                }
            )

        # Convert to qrels
        qrels = build_qrels(records=assessments, spec=CITATION_QRELS)
        print(f"Created citation qrels with {len(qrels.rows)} entries")

        return qrels

    def judge(
        self,
        rag_responses: Sequence[Report],
        rag_topics: Sequence[Request],
        llm_config: MinimaLlmConfig,
        nugget_banks: Optional[NuggetBanksProtocol] = None,
        qrels: Optional[Qrels] = None,
        **kwargs
    ) -> Leaderboard:
        """Aggregate citation assessments into leaderboard."""

        if not qrels:
            raise ValueError("CitationJudge requires qrels. Run create_qrels first.")

        # Build grade lookup: (topic_id, doc_id) -> grade
        grade_lookup = {
            (row.topic_id, row.doc_id): row.grade
            for row in qrels.rows
        }

        # Build leaderboard
        builder = LeaderboardBuilder(CITATION_SPEC)
        expected_topic_ids = [t.request_id for t in rag_topics]

        # Process each response
        for response in rag_responses:
            documents = response.documents or {}
            responses_array = response.responses or []

            # Calculate citation metrics for this response
            total_citations = 0
            existing_citations = 0
            supported_citations = 0

            for resp_segment in responses_array:
                sentence = resp_segment.text if hasattr(resp_segment, 'text') else ''
                citations_raw = resp_segment.citations if hasattr(resp_segment, 'citations') else []

                # Handle both dict (RAGTIME) and list (RAG/DRAGUN) citations
                if isinstance(citations_raw, dict):
                    # RAGTIME: citations is Dict[str, float]
                    citations = list(citations_raw.keys())
                elif citations_raw is None:
                    citations = []
                else:
                    # RAG/DRAGUN: citations is List[str] or List[int]
                    citations = [str(c) for c in citations_raw]

                total_citations += len(citations)

                for citation_id in citations:
                    # Check if exists
                    if citation_id in documents:
                        existing_citations += 1

                        # Check if supports (from qrels)
                        doc_id = doc_id_md5(f"{sentence}_{citation_id}")
                        grade = grade_lookup.get(
                            (response.metadata.topic_id, doc_id),
                            0.0
                        )
                        if grade > 0:
                            supported_citations += 1

            # Calculate metrics
            citation_accuracy = existing_citations / total_citations if total_citations > 0 else 0.0
            citation_support = supported_citations / total_citations if total_citations > 0 else 0.0
            perfect_citations = (citation_accuracy == 1.0 and citation_support == 1.0 and total_citations > 0)

            builder.add(
                run_id=response.metadata.run_id,
                topic_id=response.metadata.topic_id,
                CITATION_ACCURACY=citation_accuracy,
                CITATION_SUPPORT=citation_support,
                AVG_CITATIONS=float(total_citations),
                PERFECT_CITATIONS=perfect_citations,
            )

        leaderboard = builder.build(
            expected_topic_ids=expected_topic_ids,
            on_missing="fix_aggregate"
        )

        return leaderboard

    def create_nuggets(self, *args, **kwargs):
        """Not used by CitationJudge."""
        return None


# CLI Entry Point

if __name__ == "__main__":
    import os
    from trec_auto_judge import auto_judge_to_click_command

    # Check for debug log from environment variable (set by run_judge.py)
    debug_log = os.getenv('CITATION_DEBUG_LOG')

    judge = CitationJudge(debug_log=debug_log)
    cli = auto_judge_to_click_command(judge, "citation-judge")
    cli()
