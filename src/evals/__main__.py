"""`python -m evals` entry point (tool-calling replay).

The recall-quality harness is a sibling entry point: `python -m evals.recall`
(or `./manage eval-recall`).
"""
import sys

from .runner import main

if __name__ == "__main__":
    sys.exit(main())
