"""
Multi-topic NuggetBanks container for the Nugget Exchange system.

Provides a container that holds multiple NuggetBank instances keyed by query_id,
with support for single-bank JSON and JSONL file formats.
"""

import gzip
import json
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Union, TextIO

from pydantic import BaseModel

from .nugget_data import NuggetBank


class NuggetBanks(BaseModel):
    """
    Container for multiple NuggetBanks, keyed by query_id.

    Supports:
    - Multiple topics in single file (JSONL format)
    - Loading from directory of per-topic files
    - Iteration over topic_ids
    - Dict-like access by topic_id

    Example:
        >>> banks = NuggetBanks.empty()
        >>> banks.add_bank(my_nugget_bank)
        >>> banks["topic-001"]  # Get bank for topic
        >>> for query_id in banks:
        ...     print(query_id)
    """

    format_version: str = "v3"
    banks: Dict[str, NuggetBank] = {}

    def __getitem__(self, query_id: str) -> Optional[NuggetBank]:
        """Get NuggetBank by query_id, or None if not found."""
        return self.banks.get(query_id)

    def __contains__(self, query_id: str) -> bool:
        """Check if query_id exists in banks."""
        return query_id in self.banks

    def __iter__(self) -> Iterator[str]:
        """Iterate over query_ids."""
        return iter(self.banks)

    def __len__(self) -> int:
        """Return number of banks."""
        return len(self.banks)

    def get(self, query_id: str, default: Optional[NuggetBank] = None) -> Optional[NuggetBank]:
        """Get NuggetBank by query_id with optional default."""
        return self.banks.get(query_id, default)

    def query_ids(self) -> List[str]:
        """Return list of all query_ids."""
        return list(self.banks.keys())

    def add_bank(self, bank: NuggetBank) -> "NuggetBanks":
        """
        Add or merge a NuggetBank for its query_id.

        If a bank with the same query_id exists, nuggets are merged.
        """
        query_id = bank.query_id
        if query_id is None:
            raise ValueError("NuggetBank must have a query_id")

        if query_id in self.banks:
            existing = self.banks[query_id]
            if bank.nugget_bank:
                for nugget in bank.nugget_bank.values():
                    existing.add_nuggets(nugget)
        else:
            self.banks[query_id] = bank
        return self

    def items(self):
        """Return iterable of (query_id, NuggetBank) pairs."""
        return self.banks.items()

    def values(self):
        """Return iterable of NuggetBank instances."""
        return self.banks.values()

    @classmethod
    def empty(cls) -> "NuggetBanks":
        """Create an empty NuggetBanks container."""
        return cls(banks={})

    #  2. Single bank (legacy): {"query_id": "...", "nugget_bank": {...}}
    # Also supports .jsonl with one NuggetBank per line.
    @classmethod  
    def from_single_bank(cls, bank: NuggetBank) -> "NuggetBanks":
        """Create from a single NuggetBank (legacy compatibility)."""
        if bank.query_id is None:
            raise ValueError("NuggetBank must have a query_id")
        return cls(banks={bank.query_id: bank})

    @classmethod
    def from_banks_list(cls, banks: List[NuggetBank]) -> "NuggetBanks":
        """Create from a list of NuggetBanks."""
        result = cls.empty()
        for bank in banks:
            result.add_bank(bank)
        return result


def load_nugget_banks_from_file(source: Union[str, Path]) -> NuggetBanks:
    """
    Load NuggetBanks from a file.

    Supports:
    1. Single bank JSON: {"query_id": "...", "nugget_bank": {...}}
    2. JSONL: One NuggetBank per line

    Args:
        source: Path to JSON, JSON.gz, JSONL, or JSONL.gz file

    Returns:
        NuggetBanks container with loaded banks
    """
    path = Path(source)
    str_path = str(path).lower()
    open_fn = gzip.open if str_path.endswith(".gz") else open

    with open_fn(path, mode="rt", encoding="utf-8") as f:
        if ".jsonl" in str_path:
            return _load_jsonl(f)
        else:
            return _load_json(f)


def _load_json(f: TextIO) -> NuggetBanks:
    """Load from single-bank JSON format."""
    data = json.load(f)
    bank = NuggetBank.model_validate(data)
    return NuggetBanks.from_single_bank(bank)


def _load_jsonl(f: TextIO) -> NuggetBanks:
    """Load from JSONL format (one NuggetBank per line)."""
    banks = []
    for line in f:
        line = line.strip()
        if line:
            data = json.loads(line)
            bank = NuggetBank.model_validate(data)
            banks.append(bank)
    return NuggetBanks.from_banks_list(banks)


def load_nugget_banks_from_directory(directory: Union[str, Path]) -> NuggetBanks:
    """
    Load NuggetBanks from a directory containing per-topic files.

    Searches for: *.json, *.json.gz, *.jsonl, *.jsonl.gz files.
    Each JSON file contains a single NuggetBank.
    Each JSONL file can contain multiple NuggetBanks.

    Args:
        directory: Path to directory containing nugget bank files

    Returns:
        NuggetBanks container with all loaded banks merged
    """
    path = Path(directory)
    result = NuggetBanks.empty()

    patterns = ["*.json", "*.json.gz", "*.jsonl", "*.jsonl.gz"]
    for pattern in patterns:
        for file_path in path.glob(pattern):
            banks = load_nugget_banks_from_file(file_path)
            for bank in banks.values():
                result.add_bank(bank)

    return result


def write_nugget_banks(
    nugget_banks: NuggetBanks,
    out: Union[str, Path],
    format: str = "jsonl"
) -> None:
    """
    Write NuggetBanks to file(s).

    Args:
        nugget_banks: The NuggetBanks container to write
        out: Output path (file for jsonl, directory for directory format)
        format: One of:
            - "jsonl": One NuggetBank per line (default)
            - "directory": One file per topic in specified directory

    Examples:
        >>> write_nugget_banks(banks, "nuggets.jsonl.gz")  # JSONL
        >>> write_nugget_banks(banks, "nuggets/", format="directory")
    """
    path = Path(out)

    if format == "directory":
        path.mkdir(parents=True, exist_ok=True)
        for query_id, bank in nugget_banks.items():
            safe_id = query_id.replace("/", "_").replace("\\", "_")
            bank_path = path / f"{safe_id}.json.gz"
            _write_single_bank(bank, bank_path)
    else:  # jsonl
        str_path = str(path).lower()
        open_fn = gzip.open if str_path.endswith(".gz") else open
        with open_fn(path, mode="wt", encoding="utf-8") as f:
            for bank in nugget_banks.values():
                f.write(bank.model_dump_json(exclude_none=True) + "\n")


def _write_single_bank(bank: NuggetBank, path: Path) -> None:
    """Write a single NuggetBank to file."""
    from .nugget_data import write_nugget_bank_json
    write_nugget_bank_json(bank, path)