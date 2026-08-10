import sys
from evaluation.run_eval import run


def main():
    """Run the evaluation pipeline and exit with the appropriate code."""
    exit_code = run()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
