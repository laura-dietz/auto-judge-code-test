"""Nugget data models and I/O utilities for nugget-based evaluation."""

from .nugget_data import (
    # Core models
    NuggetBank,
    NuggetQuestion,
    NuggetClaim,
    Answer,
    Reference,
    Creator,
    Offsets,
    AggregatorType,
    # Merge functions
    merge_nugget_questions,
    merge_nugget_claims,
    # Dict conversion
    nugget_to_dict,
    question_nugget_to_dict,
    # I/O
    load_nugget_bank_json,
    write_nugget_bank_json,
    print_nugget_json,
    str_nugget_json,
)

from .nugget_banks import (
    NuggetBanks,
    load_nugget_banks_from_file,
    load_nugget_banks_from_directory,
    write_nugget_banks,
)