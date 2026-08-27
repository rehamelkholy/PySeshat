"""
Docstring-coverage audit for src/pyseshat, per
docs/development_plan_phase2.md Phase 3 item 3 ("an audit script
is a fast way to find the actual gaps rather than guessing").

Walks every public (non-underscore-prefixed) function/method
definition under src/pyseshat and flags any whose docstring has
fewer than 3 non-blank lines. This deliberately over-flags:
many one-line-docstring hits are simple property-style getters
(``n_samples``, ``success``, ``message``, ...) where a terse
docstring is already adequate and "strengthening" it would just
be padding. The list is a starting point for a human (or a
future documentation pass) to triage, not a checklist every line
of which needs a rewrite -- see docs/project_decisions.md's Phase
3/4 entry (2026-08-27) for which items from this list were judged
worth strengthening this pass and which were deliberately left as
future work.

Run

    python scripts/audit_docstrings.py
"""

import ast
import pathlib

SRC_ROOT = pathlib.Path(__file__).resolve().parent.parent / "src" / "pyseshat"


def find_thin_docstrings(min_lines=3):

    thin = []

    for path in sorted(SRC_ROOT.rglob("*.py")):

        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:
            continue

        for node in ast.walk(tree):

            if not isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef)
            ):
                continue

            if node.name.startswith("_"):
                continue

            doc = ast.get_docstring(node)

            n_lines = (
                0 if doc is None
                else len([l for l in doc.strip().splitlines() if l.strip()])
            )

            if n_lines < min_lines:
                thin.append((str(path.relative_to(SRC_ROOT.parent.parent)), node.name, n_lines))

    return thin


def main():

    thin = find_thin_docstrings()

    print(f"{len(thin)} public functions/methods with < 3 non-blank docstring lines\n")

    for f, n, l in thin:
        print(f"{f}:{n} ({l} lines)")


if __name__ == "__main__":
    main()
