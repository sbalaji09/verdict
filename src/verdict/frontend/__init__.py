"""Phase 4: verify a frontend change in a real headless browser, in
decreasing order of trust — DOM assertion, interaction drive, and
perceptual screenshot diff are all PROVEN (executed, reproducible); the
vision-intent judge is JUDGED (an opinion) and can never flip a PROVEN
result. See DESIGN.md's Phase 4 section for the full design and the
flakiness-handling discussion.
"""

from __future__ import annotations

from verdict.frontend.runner import run_frontend_checks

__all__ = ["run_frontend_checks"]
