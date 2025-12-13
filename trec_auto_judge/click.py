from pathlib import Path
from .io import load_runs_failsave

def option_rag_responses():
    import click
    class ClickRagResponses(click.ParamType):
        name = "dir"

        def convert(self, value, param, ctx):
            if not value or not Path(value).is_dir():
                self.fail(f"The directory {value} does not exist, so I can not load rag responses from this directory.", param, ctx)
            runs = load_runs_failsave(Path(value))

            if len(runs) > 0:
                return runs

            self.fail(f"{value!r} contains no rag runs.", param, ctx)

    """Rag Run directory click option."""
    def decorator(func):
        func = click.option(
            "--rag-responses",
            type=ClickRagResponses(),
            required=True,
            help="The directory that contains the rag responses to evaluate."
        )(func)

        return func

    return decorator
