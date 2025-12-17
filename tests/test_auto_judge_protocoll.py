import unittest
from trec_auto_judge import RagAutoJudge, Sequence, Report, Request, LeaderboardEntry, Dict, MeasureName, MeasureDef
from trec_auto_judge.leaderboard import MeanOfFloats
from trec_auto_judge.click import auto_judge_to_click_command
from click.testing import CliRunner
from . import TREC_25_DATA
from pathlib import Path
from tempfile import TemporaryDirectory

class NaiveJudge(RagAutoJudge):
    def judge_topic(
        self,
        rag_topic: Request,
        rag_responses: Sequence["Report"],
    ) -> Sequence[LeaderboardEntry]:
        ret = list()
        for r in rag_responses:
            ret.append(LeaderboardEntry(r["metadata"]["narrative_id"], rag_topic.request_id, {"measure-01": 1}))
        return ret

    def measures(self) -> Dict[MeasureName, MeasureDef]:
        return {
            "measure-01": MeanOfFloats()
        }

class TestAutoJudgeProtocoll(unittest.TestCase):
    def test_minimal_auto_judge(self):
        cmd = auto_judge_to_click_command(NaiveJudge(), "my-command")

        runner = CliRunner()

        with TemporaryDirectory() as tmp_dir:
            target_file = Path(tmp_dir) / "leaderboard.trec"
            result = runner.invoke(cmd, ["--rag-responses", TREC_25_DATA / "spot-check-dataset", "--output", str(target_file)])

            print(result.output)
            print(result.exception)
            self.assertIsNone(result.exception)
            self.assertEqual("", result.output)
            self.assertEqual(result.exit_code, 0)
            self.assertTrue(target_file.is_file())
            actual_leaderboard = target_file.read_text()
            self.assertIn("measure-01\t28\t1", actual_leaderboard)
            self.assertIn("measure-01\tall\t1.0", actual_leaderboard)

