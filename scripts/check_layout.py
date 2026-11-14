import re
import sys

from settings import settings

LABEL = "end:intro"
LAST_PAGE = 1


def main():
    aux = settings.PAPER_TEX.with_suffix(".aux").read_text()
    found = re.search(r"\\newlabel\{" + LABEL + r"\}\{\{[^}]*\}\{(\d+)\}", aux)
    if not found:
        sys.exit(f"no label {LABEL} in the aux file; build the paper first")
    page = int(found.group(1))
    if page > LAST_PAGE:
        sys.exit(f"the introduction ends on page {page}, not on page {LAST_PAGE}")
    print(f"the introduction ends on page {page}")


if __name__ == "__main__":
    main()
