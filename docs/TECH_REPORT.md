# TREC AutoJudge: A Framework for Automated RAG System Evaluation

## Introduction

Developing and improving Retrieval-Augmented Generation (RAG) systems requires reliable evaluation. While manual evaluation is ideal, it does not lead to reuable benchmarks. The **Auto-Judge** track is to further rapid iteration cycles of modern development, by automatically assess RAG outputs. 

The TREC AutoJudge framework provides a standardized testbed for developing, comparing, and validating auto-judge approaches. It enables researchers and practitioners to:

- Implement auto-judges using a well-defined protocol
- Evaluate their judges against ground-truth human assessments
- Compare multiple judge variants systematically
- Iterate rapidly with automated meta-evaluation

The primaty goal is to identify which Auto-Judge implementation resembles human judgments most closely. But equally important is to develop a reliable meta-evalation paradigm that steers clear of many vulnerabilities of the LLM-as-a-Judge paradigm to avoid flawed measurements.

## The AutoJudge Protocol

The framework defines an `AutoJudge` protocol with three distinct phases, each producing different evaluation artifacts:

### 1. Nugget Creation (`create_nuggets`)

Nuggets represent atomic units of information that a RAG response should contain. The nugget creation phase extracts or generates these units for each evaluation topic.

```python
def create_nuggets(
    self,
    rag_responses: Iterable[Report],
    rag_topics: Sequence[Request],
    llm_config: MinimaLlmConfig,
    nugget_banks: Optional[NuggetBanksProtocol] = None,
    **kwargs
) -> Optional[NuggetBanksProtocol]:
    ...
```

Nuggets can be:
- **Questions with answers** (`NuggetQuestion`): What should a good response address?
- **Claims** (`NuggetClaim`): What facts should a good response contain?

Nugget creation can be parametric (from LLM knowledge) or grounded in source passages.

### 2. Qrels Creation (`create_qrels`)

Qrels (relevance judgments) provide fine-grained assessments of individual response components. This phase produces document-level or passage-level relevance grades.

```python
def create_qrels(
    self,
    rag_responses: Iterable[Report],
    rag_topics: Sequence[Request],
    llm_config: MinimaLlmConfig,
    nugget_banks: Optional[NuggetBanksProtocol] = None,
    **kwargs
) -> Optional[Qrels]:
    ...
```

Qrels enable evaluation with standard IR metrics and facilitate deeper analysis of where RAG systems succeed or fail.

### 3. Judging (`judge`)

The judge phase produces a leaderboard ranking RAG systems by quality. This is the primary output for most evaluation scenarios.

```python
def judge(
    self,
    rag_responses: Iterable[Report],
    rag_topics: Sequence[Request],
    llm_config: MinimaLlmConfig,
    nugget_banks: Optional[NuggetBanksProtocol] = None,
    qrels: Optional[Qrels] = None,
    **kwargs
) -> Leaderboard:
    ...
```

The leaderboard contains scores per run and topic, with configurable measures defined by a schema (`LeaderboardSpec`).

### Modular Design

These three phases can be implemented by a single class or composed from separate implementations. For example:
- `RubricJudge` creates nuggets; `PrefNuggetJudge` performs judging
- A single `UmbrelaJudge` handles all phases internally

## Input Data: Topics and RAG Reports

### Topics (Requests)

Topics define what the RAG systems are asked to address:

```python
Request(
    request_id="topic-001",
    title="Climate Change Effects",
    problem_statement="Explain the effects of climate change on coastal cities",
    background="User is a policy researcher...",
)
```

Topics can be loaded from IR datasets or JSONL files.

### RAG Reports

RAG system outputs are provided as `Report` objects:

```python
Report(
    team_id="team-xyz",
    run_id="run-001",
    topic_id="topic-001",
    responses=[...],  # System's generated response(s)
)
```

Reports contain the RAG system's answer to a topic, including any citations or supporting passages.

## Workflow Configuration

Auto-judges are configured via `workflow.yml` files that control:

### Execution Phases

```yaml
create_nuggets: true    # Generate nuggets
judge: true             # Produce leaderboard
judge_uses_nuggets: true
```

### Class Configuration

```yaml
# Single class handles everything
auto_judge: "trec25.judges.umbrela:UmbrelaJudge"

# Or separate classes for each phase
nugget_class: "trec25.judges.rubric:RubricJudge"
judge_class: "trec25.judges.prefnugget:PrefNuggetJudge"
```

### Settings and Parameters

```yaml
nugget_settings:
  prompt: "prefnugget-baseline"
  max_nuggets_per_topic: 20

judge_settings:
  grade_range: [0, 3]
```

### Variants and Sweeps

The framework supports systematic experimentation:

- **Default**: The standard configuration for a judge
- **Variants**: Named alternative configurations (e.g., different prompts)
- **Sweeps**: Systematic parameter exploration (e.g., varying temperature)

Each configuration produces a distinct auto-judge system that can be evaluated independently.

## Meta-Evaluation: Correlation Analysis

The core question: *Does the auto-judge rank systems the same way humans do?*

Meta-evaluation computes correlations between auto-judge leaderboards and ground-truth human assessments.

### Leaderboard Correlations

| Metric | Description |
|--------|-------------|
| Kendall's τ | Rank correlation based on concordant/discordant pairs |
| Spearman's ρ | Rank correlation using rank differences |
| Pearson's r | Linear correlation of raw scores |
| τAP | AP-weighted Kendall correlation |

### Top-k Correlations

Often we care most about ranking the *best* systems correctly. The framework provides `correlation@k` metrics:

- `kendall@10`: Correlation among the top 10 systems
- `spearman@5`: Correlation among the top 5 systems

These metrics reveal whether an auto-judge correctly identifies which RAG systems are strongest, even if it struggles with mid-tier rankings.

### Label Correlations (Coming Soon)

For finer-grained evaluation at the judgment level:

| Metric | Description |
|--------|-------------|
| Cohen's κ | Inter-rater agreement for categorical labels |
| Krippendorff's α | Agreement measure handling missing data |
| Overlap/Jaccard | Set similarity for binary relevance |

## The `meta-evaluate` Command

```bash
trec-auto-judge meta-evaluate \
    --truth-leaderboard official.eval.jsonl \
    --truth-format jsonl \
    --input autojudge-*.txt \
    --eval-format tot \
    --correlation kendall \
    --correlation spearman \
    --correlation kendall@10 \
    --output correlations.jsonl
```

### Key Features

- **Multiple input files**: Evaluate multiple auto-judge variants at once
- **Flexible formats**: JSONL, TOT, ir_measures
- **Run/topic filtering**: `--only-shared-runs`, `--only-shared-topics`
- **Missing value handling**: `--on-missing {error|warn|skip|default}`
- **Multiple measures**: Compare across different evaluation measures

## Automated Watchdog

For continuous evaluation during development, `meta-watch.sh` provides:

- **Change detection**: Monitors for new evaluation files via rsync
- **Automatic correlation**: Runs meta-evaluate when new results arrive
- **Result publishing**: Syncs correlation results to a destination

```bash
watch -n 60 ./meta-watch.sh /data/truth server:/in server:/out
```

This enables a rapid feedback loop: submit auto-judge results → receive correlation analysis.

> Note: The watchdog will be replaced with TIRA integration for production deployments.

---

## Examples

*[Examples from real data to follow]*
