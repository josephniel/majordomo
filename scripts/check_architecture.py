#!/usr/bin/env python
"""Run the import-linter contracts and print the report.

Wraps the library rather than shelling out to `lint-imports`: that CLI
produces no output under this project's environment, which would make a
broken contract look like a passing build — the exact failure mode these
contracts exist to prevent.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from importlinter.cli import lint_imports  # noqa: E402

if __name__ == "__main__":
    sys.exit(lint_imports(verbose=True))
