import sys
from evaluation.run_eval import run


def main():
    exit_code = run()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
