#!/usr/bin/env python3
from trec_auto_judge import option_rag_responses, Report, LeaderboardSpec, LeaderboardBuilder, mean_of_floats, MeasureSpec
import click
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from statistics import mean
import random


def rand(seed: str) -> float:
    random.seed(seed)
    return random.random()


NAIVE_LEADERBOARD_SPEC = LeaderboardSpec(measures=(
    MeasureSpec("LENGTH", aggregate=mean_of_floats, cast=float),
    MeasureSpec("RANDOM", aggregate=mean_of_floats, cast=float),
))


@click.command("naive_baseline")
@click.option("--output", type=Path, help="The output file.", required=True)
@option_rag_responses()
def main(rag_responses: list[Report], output: Path):
    """
    A naive rag response assessor that just orders each response by its length.
    """
    ret = LeaderboardBuilder(NAIVE_LEADERBOARD_SPEC)

    for rag_response in tqdm(rag_responses, "Process RAG Responses"):
        vals = {
            "LENGTH": len(rag_response.get_report_text().split()),
            "RANDOM": rand(rag_response.metadata.run_id + rag_response.metadata.topic_id)
        }
        ret.add(run_id=rag_response.metadata.run_id, topic_id=rag_response.metadata.topic_id, values=vals)

    ret.build().write(output=output)


if __name__ == '__main__':
    main()