"""Interactive-program blocklist: refuse TTY-dependent commands before execution."""

from __future__ import annotations

import os
import re
import shlex

BLOCKED: frozenset[str] = frozenset({
    "vim", "vi", "nano", "emacs",
    "less", "more",
    "top", "htop",
    "man",
    "ssh",
    "tmux", "screen",
    "watch",
    "tail",
    "python", "python3",
    "node",
    "irb",
    "psql", "mysql", "sqlite3",
})

# Operators that separate pipeline segments
_SEGMENT_SPLIT = re.compile(r"\|\|?|&&|;")


def check_blocked(command: str) -> str | None:
    """Return the first blocked program name found in any pipeline segment, or None.

    Splits on |, ||, &&, ; to catch blocked programs in any position of a compound
    command. Uses shlex to parse each segment's leading token.
    """
    segments = _SEGMENT_SPLIT.split(command)
    for segment in segments:
        segment = segment.strip()
        if not segment:
            continue
        try:
            tokens = shlex.split(segment)
        except ValueError:
            continue
        if not tokens:
            continue
        prog = os.path.basename(tokens[0])
        if prog in BLOCKED:
            return prog
    return None
