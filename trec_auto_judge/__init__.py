from click import group
from .report import Report, load_report
from .request import Request, load_requests_from_irds, load_requests
from ._commands._evaluate import evaluate

__version__ = '0.0.1'

@group()
def main():
    pass


main.command()(evaluate)


if __name__ == '__main__':
    main()
