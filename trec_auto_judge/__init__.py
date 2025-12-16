from click import group
<<<<<<< HEAD
from .request import Request, load_requests_from_irds, load_requests_from_file
=======
from .report import Report, load_report
from .request import Request, load_requests
>>>>>>> 0a2139c (intermittent changes trying to get new TIRA working)
from ._commands._evaluate import evaluate
from ._commands._export_corpus import export_corpus

__version__ = '0.0.1'


@group()
def main():
    pass


main.command()(evaluate)
main.command()(export_corpus)


if __name__ == '__main__':
    main()
