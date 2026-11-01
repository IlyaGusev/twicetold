import ast
import io
import re
import sys
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKIP_DIRS = {".venv", ".git", ".claude", "data", "logs", "reports", "paper", "__pycache__"}
HASH_COMMENT_SUFFIXES = {".sh", ".yml", ".yaml", ".toml", ".cfg", ".ini"}
DOCSTRING_OWNERS = (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
QUOTED_TEXT = re.compile(r"'[^']*'|\"(?:\\.|[^\"\\])*\"")
HASH_COMMENT = re.compile(r"(^|\s)#")


def is_shebang(token):
    return token.start == (1, 0) and token.string.startswith("#!")


def python_findings(path):
    source = path.read_text()
    tokens = tokenize.generate_tokens(io.StringIO(source).readline)
    findings = [(token.start[0], "comment") for token in tokens if token.type == tokenize.COMMENT and not is_shebang(token)]
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, DOCSTRING_OWNERS) and ast.get_docstring(node, clean=False) is not None:
            findings.append((node.body[0].lineno, "docstring"))
    return findings


def hash_findings(path):
    findings = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if line_number == 1 and line.startswith("#!"):
            continue
        if HASH_COMMENT.search(QUOTED_TEXT.sub("''", line)):
            findings.append((line_number, "comment"))
    return findings


def jinja_findings(path):
    return [(line_number, "comment") for line_number, line in enumerate(path.read_text().splitlines(), 1) if "{#" in line]


FINDERS_BY_SUFFIX = {".py": python_findings, ".j2": jinja_findings} | dict.fromkeys(HASH_COMMENT_SUFFIXES, hash_findings)


def main():
    findings = []
    for path in sorted(ROOT.rglob("*")):
        relative_path = path.relative_to(ROOT)
        find_comments = FINDERS_BY_SUFFIX.get(path.suffix)
        if find_comments is None or not path.is_file() or SKIP_DIRS & set(relative_path.parts):
            continue
        findings += [(relative_path, line, kind) for line, kind in find_comments(path)]
    for path, line, kind in findings:
        print(f"{path}:{line}: {kind}")
    if findings:
        print(f"{len(findings)} comments or docstrings found; this repository allows none")
        sys.exit(1)
    print("no comments or docstrings")


if __name__ == "__main__":
    main()
