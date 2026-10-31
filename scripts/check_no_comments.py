import ast
import io
import re
import sys
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKIP_DIRS = {".venv", ".git", "data", "logs", "reports", "paper", "__pycache__"}
SKIP_FILES = {"requirements.txt", "uv.lock"}
HASH_COMMENT_SUFFIXES = {".sh", ".yml", ".yaml", ".toml", ".cfg", ".ini"}


def python_findings(path):
    src = path.read_text()
    out = []
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type == tokenize.COMMENT:
            out.append((tok.start[0], "comment"))
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
                out.append((body[0].lineno, "docstring"))
    return out


def hash_findings(path):
    out = []
    for i, line in enumerate(path.read_text().splitlines(), 1):
        if i == 1 and line.startswith("#!"):
            continue
        if re.match(r"\s*#", line) or re.search(r"\s#\s", line):
            out.append((i, "comment"))
    return out


def jinja_findings(path):
    return [(i, "comment") for i, line in enumerate(path.read_text().splitlines(), 1) if "{#" in line]


def main():
    findings = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path.name in SKIP_FILES or SKIP_DIRS & set(path.relative_to(ROOT).parts):
            continue
        if path.suffix == ".py":
            found = python_findings(path)
        elif path.suffix in HASH_COMMENT_SUFFIXES:
            found = hash_findings(path)
        elif path.suffix == ".j2":
            found = jinja_findings(path)
        else:
            continue
        findings += [(path.relative_to(ROOT), line, kind) for line, kind in found]
    for path, line, kind in findings:
        print(f"{path}:{line}: {kind}")
    if findings:
        print(f"{len(findings)} comments or docstrings found; this repository allows none")
        sys.exit(1)
    print("no comments or docstrings")


if __name__ == "__main__":
    main()
