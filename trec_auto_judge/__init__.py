from click import group
from .report import Report, load_report
from .request import Request, load_requests_from_irds, load_requests
from .leaderboard import Leaderboard, LeaderboardEntry, MeasureDef, MeasureName, write_leaderboard
from .qrels import Qrels, QrelEntry, write_qrel_file, read_qrel_file
from ._commands._evaluate import evaluate

__version__ = '0.0.1'


@group()
def main():
    pass


main.command()(evaluate)


if __name__ == '__main__':
    main()
