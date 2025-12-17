
from pathlib import Path
from collections import defaultdict
from statistics import mean
from dataclasses import dataclass, field

from typing import Union, Callable, Mapping, Sequence, Dict, List, DefaultDict, Iterable, Any
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple


MeasureName = str


@dataclass(frozen=True)
class LeaderboardEntry:
    run_id: str
    topic_id: str
    values: Dict[MeasureName, Any]


@dataclass(frozen=True)
class Leaderboard:
    """
    Thin serialization vessel.
    """
    measures: Tuple[MeasureName, ...]     # schema only
    entries: Tuple[LeaderboardEntry, ...]
    all_topic_id: str = "all"

    def all_measure_names(self):
        return self.measures

    def write(self, output: Path) -> None:
        """
        Write the leaderboard to a file.
        """
        lines: List[str] = []
        for e in self.entries:
            for m in self.all_measure_names():
                if m in e.values:
                    lines.append("\t".join([
                        e.run_id,
                        m,
                        e.topic_id,
                        str(e.values[m]),
                    ]))

        output.parent.mkdir(parents=True, exist_ok=True)
        print(f"Writing leaderboard to {output.absolute()}")
        output.write_text("\n".join(lines) + "\n", encoding="utf-8")

# =========

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence


AggFn = Callable[[Sequence[Any]], Any]
CastFn = Callable[[Any], Any]


@dataclass(frozen=True)
class MeasureSpec:
    name: MeasureName
    aggregate: AggFn
    cast: CastFn = lambda x: x


@dataclass(frozen=True)
class LeaderboardSpec:
    measures: Tuple[MeasureSpec, ...]
    all_topic_id: str = "all"

    @property
    def names(self) -> Tuple[MeasureName, ...]:
        return tuple(m.name for m in self.measures)

    @property
    def name_set(self) -> set[MeasureName]:
        return set(self.names)

    def cast_values(self, values: Mapping[MeasureName, Any]) -> Dict[MeasureName, Any]:
        return {m.name: m.cast(values[m.name]) for m in self.measures}
# =====

from collections import defaultdict
from typing import Iterable, Callable, Optional, List


class LeaderboardBuilder:
    def __init__(self, spec: LeaderboardSpec):
        self.spec = spec
        self._rows: List[LeaderboardEntry] = []

    def add(
        self,
        *,
        run_id: str,
        topic_id: str,
        values: Optional[Dict[MeasureName, Any]] = None,
        **kw: Any,
    ) -> None:
        if values is None:
            values = kw
        elif kw:
            raise TypeError("Pass either values= or keyword measures, not both.")

        extra = set(values) - self.spec.name_set
        missing = self.spec.name_set - set(values)
        if extra:
            raise KeyError(f"Unknown measure(s): {sorted(extra)}")
        if missing:
            raise KeyError(f"Missing measure(s): {sorted(missing)}")

        casted = self.spec.cast_values(values)
        self._rows.append(LeaderboardEntry(run_id, topic_id, casted))

    def add_records(
        self,
        records: Iterable[Any],
        *,
        run_id: Callable[[Any], str],
        topic_id: Callable[[Any], str],
        get_values: Callable[[Any], Dict[MeasureName, Any]],
    ) -> None:
        for r in records:
            self.add(
                run_id=run_id(r),
                topic_id=topic_id(r),
                values=get_values(r),
            )

    def build(self) -> Leaderboard:
        by_run = defaultdict(lambda: defaultdict(list))

        for e in self._rows:
            if e.topic_id == self.spec.all_topic_id:
                continue
            for k, v in e.values.items():
                by_run[e.run_id][k].append(v)

        all_rows: List[LeaderboardEntry] = []
        for run_id, m2vals in by_run.items():
            agg_vals: Dict[MeasureName, Any] = {}
            for ms in self.spec.measures:
                vals = m2vals.get(ms.name, [])
                if vals:
                    agg_vals[ms.name] = ms.aggregate(vals)
            all_rows.append(
                LeaderboardEntry(run_id, self.spec.all_topic_id, agg_vals)
            )

        return Leaderboard(
            measures=self.spec.names,
            entries=tuple(self._rows + all_rows),
            all_topic_id=self.spec.all_topic_id,
        )

# == Aggregator ==


def mean_of_floats(values):
    return mean(float(v) for v in values)


def mean_of_bools(values):
    return mean(1.0 if bool(v) else 0.0 for v in values)