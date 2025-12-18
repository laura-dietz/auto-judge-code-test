import click
from pathlib import Path
from ..io import load_hf_dataset_config_or_none


@click.argument("corpus-directory", type=Path)
def export_corpus(corpus_directory: Path) -> int:
    """Export a corpus into a pre-defined directory so that everything is self-contained."""
    import ir_datasets
    from tira.ir_datasets_loader import IrDatasetsLoader
    # todo create a shared method
    irds_id = load_hf_dataset_config_or_none(corpus_directory / "README.md", ["ir_dataset"])["ir_dataset"]["ir_datasets_id"]
    ds = ir_datasets.load(irds_id)
    irds_loader = IrDatasetsLoader()
    irds_loader.load_dataset_for_fullrank(
        irds_id,
        corpus_directory,
        output_dataset_truth_path=None,
        skip_documents=True,
        skip_qrels=True
    )
    return 0