import re
from typing import Optional, Type, Sequence
from collections import Counter, defaultdict
from tqdm import tqdm
from pydantic import BaseModel

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
    build_qrels,
    doc_id_md5,
)
from trec_auto_judge.leaderboard.leaderboard import mean_of_floats
from trec_auto_judge.nugget_data import NuggetBanks, NuggetBanksProtocol


# ============================================================================
# Data Model
# ============================================================================

class NonLlmScore(BaseModel):
    run_id: str
    topic_id: str
    passage: str
    query: str

    length_score: float = 0.0
    keyword_score: float = 0.0
    bm25_score: float = 0.0
    coverage_score: float = 0.0
    combined_score: float = 0.0
    grade: int = 0


# ============================================================================
# Specs
# ============================================================================

NON_LLM_SPEC = LeaderboardSpec(measures=(
    MeasureSpec("AVG_GRADE", aggregate=mean_of_floats, cast=float, default=0.0),
))

NON_LLM_QRELS = QrelsSpec[NonLlmScore](
    topic_id=lambda r: r.topic_id,
    doc_id=lambda r: doc_id_md5(r.passage),
    grade=lambda r: float(r.grade),
    on_duplicate="keep_max"
)


# ============================================================================
# Judge
# ============================================================================

class NonLlmJudge(AutoJudge):

    nugget_banks_type: Type[NuggetBanksProtocol] = NuggetBanks

    def __init__(self, settings: Optional[dict] = None, **kwargs):
        super().__init__(settings=settings, **kwargs)

        self.length_weight = settings.get("length_weight", 0.2) if settings else 0.2
        self.keyword_weight = settings.get("keyword_weight", 0.3) if settings else 0.3
        self.bm25_weight = settings.get("bm25_weight", 0.3) if settings else 0.3
        self.coverage_weight = settings.get("coverage_weight", 0.2) if settings else 0.2

        # normalize weights
        total = self.length_weight + self.keyword_weight + self.bm25_weight + self.coverage_weight
        self.length_weight /= total
        self.keyword_weight /= total
        self.bm25_weight /= total
        self.coverage_weight /= total

        self.min_length = settings.get("min_length", 50) if settings else 50
        self.optimal_length = settings.get("optimal_length", 200) if settings else 200
        self.max_length = settings.get("max_length", 500) if settings else 500

        self.k1 = 1.5
        self.b = 0.75

        # Topic format: auto-detect or explicit
        self.topic_format = settings.get("topic_format", "auto") if settings else "auto"

    # ------------------------------------------------------------------------

    def extract_query(self, topic) -> str:
        """
        Extract query text from topic based on format.
        Supports: RAGTIME (title+problem+background), DRAGUN (body), RAG (title), auto-detect
        """
        # Explicit format override
        if self.topic_format == "dragun":
            return getattr(topic, "body", topic.title)
        elif self.topic_format == "rag":
            return topic.title
        elif self.topic_format == "ragtime":
            parts = [topic.title, getattr(topic, "problem_statement", ""), getattr(topic, "background", "")]
            return " ".join(p for p in parts if p).strip()

        # Auto-detect format
        if hasattr(topic, 'body') and topic.body:  # DRAGUN
            return topic.body
        elif hasattr(topic, 'problem_statement'):  # RAGTIME
            parts = [topic.title, getattr(topic, "problem_statement", ""), getattr(topic, "background", "")]
            return " ".join(p for p in parts if p).strip()
        else:  # RAG (simple title-only)
            return topic.title

    # ------------------------------------------------------------------------

    def tokenize(self, text: str) -> list[str]:
        return re.findall(r"\b\w+\b", text.lower())

    # ------------------------------------------------------------------------

    def compute_length_score(self, response: str) -> float:
        wc = len(self.tokenize(response))

        if wc < self.min_length:
            score = wc / self.min_length * 0.5
        elif wc <= self.optimal_length:
            score = 0.5 + (wc - self.min_length) / (self.optimal_length - self.min_length) * 0.5
        elif wc <= self.max_length:
            excess = (wc - self.optimal_length) / (self.max_length - self.optimal_length)
            score = 1.0 - excess * 0.2
        else:
            score = 0.8 * (self.max_length / wc)

        return max(0.0, min(1.0, score))

    # ------------------------------------------------------------------------

    def compute_keyword_score(self, query: str, response: str) -> float:
        q = set(self.tokenize(query))
        r = set(self.tokenize(response))
        if not q:
            return 0.0
        return len(q & r) / len(q)

    # ------------------------------------------------------------------------

    def compute_query_coverage(self, query: str, response: str) -> float:
        q_tokens = self.tokenize(query)
        r_counter = Counter(self.tokenize(response))
        q_counter = Counter(q_tokens)

        if not q_tokens:
            return 0.0

        covered = sum(min(r_counter[t], q_counter[t]) for t in q_counter)
        return covered / sum(q_counter.values())

    # ------------------------------------------------------------------------

    def compute_bm25_score(self, query: str, response: str) -> float:
        q_tokens = self.tokenize(query)
        d_tokens = self.tokenize(response)

        if not q_tokens or not d_tokens:
            return 0.0

        doc_len = len(d_tokens)
        avgdl = self.optimal_length

        tf = Counter(d_tokens)
        score = 0.0

        for t in q_tokens:
            if t in tf:
                f = tf[t]
                denom = f + self.k1 * (1 - self.b + self.b * doc_len / avgdl)
                score += (f * (self.k1 + 1)) / denom

        max_score = len(set(q_tokens)) * (self.k1 + 1)
        return min(score / max_score, 1.0) if max_score else 0.0

    # ------------------------------------------------------------------------

    def compute_combined_score(self, query: str, response: str) -> NonLlmScore:
        ls = self.compute_length_score(response)
        ks = self.compute_keyword_score(query, response)
        bs = self.compute_bm25_score(query, response)
        cs = self.compute_query_coverage(query, response)

        combined = (
            ls * self.length_weight +
            ks * self.keyword_weight +
            bs * self.bm25_weight +
            cs * self.coverage_weight
        )

        final = combined * 3.0
        grade = int(round(final))
        grade = max(0, min(3, grade))

        return NonLlmScore(
            run_id="",
            topic_id="",
            passage=response,
            query=query,
            length_score=ls,
            keyword_score=ks,
            bm25_score=bs,
            coverage_score=cs,
            combined_score=final,
            grade=grade,
        )

    # ------------------------------------------------------------------------

    def create_qrels(self, rag_responses, rag_topics, llm_config, nugget_banks=None, **kwargs) -> Optional[Qrels]:
        topic_dict = {req.request_id: req for req in rag_topics}

        scores = []
        for response in tqdm(rag_responses, "Scoring responses"):
            topic = topic_dict.get(response.metadata.topic_id)
            if not topic:
                continue

            query = self.extract_query(topic)

            # Use get_report_text() to combine all sentences into one response
            full_response = response.get_report_text()
            score = self.compute_combined_score(query, full_response)
            score.run_id = response.metadata.run_id
            score.topic_id = response.metadata.topic_id
            scores.append(score)

        return build_qrels(records=scores, spec=NON_LLM_QRELS)

    # ------------------------------------------------------------------------

    def judge(self, rag_responses, rag_topics, llm_config, nugget_banks=None, **kwargs) -> Leaderboard:
        topic_dict = {req.request_id: req for req in rag_topics}
        all_scores = {}

        for response in tqdm(rag_responses, "Building leaderboard"):
            topic = topic_dict.get(response.metadata.topic_id)
            if not topic:
                continue

            query = self.extract_query(topic)

            # Use get_report_text() to combine all sentences into one response
            full_response = response.get_report_text()
            score = self.compute_combined_score(query, full_response)
            all_scores[(response.metadata.run_id, response.metadata.topic_id)] = score.grade

        ret = LeaderboardBuilder(NON_LLM_SPEC)

        # Add per-topic scores
        for (run_id, topic_id), grade in all_scores.items():
            ret.add(run_id=run_id, topic_id=topic_id, values={"AVG_GRADE": grade})

        # Add aggregate scores
        run_scores = defaultdict(list)
        for (run_id, _), grade in all_scores.items():
            run_scores[run_id].append(grade)

        for run_id, vals in run_scores.items():
            ret.add(run_id=run_id, topic_id="all", values={"AVG_GRADE": sum(vals) / len(vals)})

        return ret.build()


if __name__ == "__main__":
    from trec_auto_judge import auto_judge_to_click_command

    judge = NonLlmJudge()
    auto_judge_to_click_command(judge, "non_llm_judge")()
