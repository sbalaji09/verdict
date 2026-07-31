"""The subprocess entrypoint `git bisect run` invokes at every candidate
commit: `python -m verdict.attribution.bisect_cli <gate> [identity]`.

`git bisect` checks out each candidate commit directly into the current
working directory rather than copying anywhere, so this just checks
whatever's in `Path.cwd()` — no path argument needed. Exit code is how
`git bisect run` receives the verdict: 0 = good, 1 = bad, 125 = skip (its
reserved "untestable, route around this commit" code).
"""

from __future__ import annotations

import sys
from pathlib import Path

from verdict.attribution.reproduce import Reproduction, check_reproduces

_EXIT_CODES = {
    Reproduction.GOOD: 0,
    Reproduction.BAD: 1,
    Reproduction.SKIP: 125,
}


def main(argv: list[str]) -> int:
    if not argv:
        return 125
    gate = argv[0]
    identity = argv[1] if len(argv) > 1 and argv[1] else None
    result = check_reproduces(gate, identity, Path.cwd())
    return _EXIT_CODES[result]


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
