from click import group
from typing import Protocol, Sequence, Optional, Dict, Union, runtime_checkable

from .report import Report, load_report
from .request import Request, load_requests_from_irds, load_requests_from_file
from .leaderboard import Leaderboard, LeaderboardEntry, MeasureDef, MeasureName
from .qrels import Qrels, QrelEntry, write_qrel_file, read_qrel_file
from ._commands._evaluate import evaluate

__version__ = '0.0.1'


@group()
def main():
    pass


main.command()(evaluate)


@runtime_checkable
class RagAutoJudge(Protocol):
    def judge(
        self,
        rag_responses: Sequence["Report"],
        rag_topics: Sequence["Request"],
    ) -> Leaderboard:
        ret: Sequence[LeaderboardEntry] = list()
        all_topics = list(rag_topics)
        for topic in all_topics:
            responses_for_topic = [i for i in rag_responses if str(i["metadata"]["narrative_id"]) == str(topic.request_id)]
            if len(responses_for_topic) == 0:
                continue

            ret += list(self.judge_topic(topic, responses_for_topic))

        return Leaderboard.from_entries_with_all(entries=ret, measures=self.measures())

    def judge_topic(
        self,
        rag_topic: Request,
        rag_responses: Sequence["Report"],
    ) -> Sequence[LeaderboardEntry]:
        ...

    def measures(self) -> Dict[MeasureName, MeasureDef]:
        ...

@runtime_checkable
class QrelAutoJudge(Protocol):
    def judge_qrels(
        self,
        rag_responses: Sequence["Report"],
        rag_topics: Sequence["Request"],
    ) -> Qrels:
        ...

AutoJudge = Union[RagAutoJudge, QrelAutoJudge]

if __name__ == '__main__':
    main()
