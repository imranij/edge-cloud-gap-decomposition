"""Fingerprint of Python code that ignores comments, docstrings and layout.

The fingerprint is the SHA-256 sum of the abstract syntax tree of a file after every docstring has been
removed. Comments, blank lines and line breaks do not enter the tree, so two files with the same fingerprint
run the same code even when their comments or docstrings differ. It is used to show that the files listed in
plan/postplan/HASHES.txt, whose comments and docstrings were edited for this repository, are the code that was
hashed before the runs.

    python tools/code_fingerprint.py src/sens/simulation_full/*.py
    python tools/code_fingerprint.py --check plan/postplan/CODE_FINGERPRINTS.txt
"""
from __future__ import annotations

import ast
import hashlib
import sys
from pathlib import Path


def _strip_docstrings(tree: ast.AST) -> ast.AST:
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
                    and isinstance(body[0].value.value, str):
                node.body = body[1:] or [ast.Pass()]
    return tree


def fingerprint(path: str | Path) -> str:
    tree = _strip_docstrings(ast.parse(Path(path).read_text(encoding="utf-8")))
    return hashlib.sha256(ast.dump(tree, include_attributes=False).encode("utf-8")).hexdigest()


def main(argv: list[str]) -> int:
    if argv[:1] == ["--check"]:
        root = Path(argv[1]).resolve().parents[2]
        bad = 0
        for line in Path(argv[1]).read_text(encoding="utf-8").splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            expected, rel = line.split(maxsplit=1)
            got = fingerprint(root / rel)
            ok = got == expected
            bad += not ok
            print(("OK      " if ok else "DIFFERS ") + rel)
        return 1 if bad else 0
    for p in argv:
        print(fingerprint(p), p)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
