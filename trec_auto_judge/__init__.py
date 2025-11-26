from click import group
from ._commands._evaluate import evaluate

__version__ = '0.0.1'

@group()
def main():
    pass


main.command()(evaluate)


if __name__ == '__main__':
    main()
