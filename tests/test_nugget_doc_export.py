"""Smoke tests for nugget-relevant document export functions."""

import json
from pathlib import Path

from trec25.judges.shared.rubric_common import (
    NuggetGradeData,
    collect_nugget_relevant_docs,
    nugget_docs_to_nugget_banks,
    write_nugget_docs_collaborator,
)


def _make_grade_data() -> list[NuggetGradeData]:
    """Create minimal grade data spanning two topics and two runs."""
    return [
        # Topic t1, two docs, one above threshold
        NuggetGradeData(run_id="r1", query_id="t1", nugget_id="n1",
                        question="What is X?", passage="p", doc_id="d1", grade=4),
        NuggetGradeData(run_id="r1", query_id="t1", nugget_id="n1",
                        question="What is X?", passage="p", doc_id="d2", grade=1),
        # Same doc from different run — should deduplicate
        NuggetGradeData(run_id="r2", query_id="t1", nugget_id="n1",
                        question="What is X?", passage="p", doc_id="d1", grade=5),
        # Topic t2
        NuggetGradeData(run_id="r1", query_id="t2", nugget_id="n2",
                        question="What is Y?", passage="p", doc_id="d3", grade=3),
        # No doc_id — should be skipped
        NuggetGradeData(run_id="r1", query_id="t1", nugget_id="n1",
                        question="What is X?", passage="p", grade=5),
    ]


def test_collect_nugget_relevant_docs():
    topics = collect_nugget_relevant_docs(_make_grade_data(), grade_threshold=3)
    assert set(topics.keys()) == {"t1", "t2"}
    # t1: only d1 meets threshold (d2 has grade 1)
    t1_docs = topics["t1"].entries[0].doc_ids
    assert t1_docs == ["d1"]
    # t2: d3 meets threshold
    assert topics["t2"].entries[0].doc_ids == ["d3"]


def test_write_nugget_docs_collaborator(tmp_path: Path):
    topics = collect_nugget_relevant_docs(_make_grade_data())
    write_nugget_docs_collaborator(topics, tmp_path)

    path = tmp_path / "nuggets_t1.json"
    assert path.exists()
    data = json.loads(path.read_text())
    assert "What is X?" in data
    assert data["What is X?"][0] == "OR"


def test_nugget_docs_to_nugget_banks():
    topics = collect_nugget_relevant_docs(_make_grade_data())
    banks = nugget_docs_to_nugget_banks(topics)
    assert "t1" in banks.banks
    assert "t2" in banks.banks
    nuggets = banks.banks["t1"].nuggets_as_list()
    assert len(nuggets) == 1
    assert nuggets[0].question == "What is X?"
    assert len(nuggets[0].references) == 1
    assert nuggets[0].references[0].doc_id == "d1"
