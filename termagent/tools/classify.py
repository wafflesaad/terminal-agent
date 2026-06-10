"""Command classifier: allowlist-based, fail-closed. AUTO only for known read-only commands."""

from __future__ import annotations

import os
import re
import shlex
from enum import Enum


class Decision(Enum):
    AUTO = "auto"
    CONFIRM = "confirm"


READ_ONLY: frozenset[str] = frozenset({
    "ls", "pwd", "cat", "head", "tail", "wc", "stat", "file", "tree", "du", "df", "find",
    "grep", "rg", "egrep", "fgrep", "which", "whereis", "type", "echo", "printf",
    "ps", "env", "printenv", "date", "whoami", "id", "uname", "hostname", "uptime",
    "git", "sort", "uniq", "cut", "tr", "diff", "cmp", "sed", "awk", "jq", "column",
    "basename", "dirname",
})

_SEGMENT_SPLIT = re.compile(r"\|\|?|&&|;|\n")
_ENV_ASSIGN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")

_GIT_READ_SUBCOMMANDS: frozenset[str] = frozenset({
    "status", "log", "diff", "show", "branch", "remote",
})


def _git_is_read_only(tokens: list[str]) -> bool:
    if len(tokens) < 2:
        return False
    subcommand = tokens[1]
    return subcommand in _GIT_READ_SUBCOMMANDS


def _find_is_read_only(tokens: list[str]) -> bool:
    mutating = {"-delete", "-exec", "-execdir", "-ok", "-okdir"}
    return not any(t in mutating for t in tokens)


def _sed_is_read_only(tokens: list[str]) -> bool:
    for t in tokens[1:]:
        if t == "-i" or t == "--in-place":
            return False
        # -i.bak or -ibak style
        if t.startswith("-i") and len(t) > 2 and t[2] != " ":
            return False
    return True


def _sort_is_read_only(tokens: list[str]) -> bool:
    for t in tokens[1:]:
        if t in ("-o", "--output"):
            return False
        # --output=file
        if t.startswith("--output="):
            return False
    return True


def _segment_is_auto(segment: str, strict: bool) -> bool:
    segment = segment.strip()
    if not segment:
        return False

    # Redirection check: > or >> or 2> writes to disk
    if ">" in segment:
        return False

    try:
        tokens = shlex.split(segment)
    except ValueError:
        return False

    if not tokens:
        return False

    # Strip leading env assignments (FOO=bar cmd)
    while tokens and _ENV_ASSIGN.match(tokens[0]):
        tokens = tokens[1:]

    if not tokens:
        return False

    base = os.path.basename(tokens[0])

    # sudo / privilege escalation
    if base in ("sudo", "doas"):
        return False

    if base not in READ_ONLY:
        return False

    # Mutating-flag sub-checks for "mostly-read" commands
    if strict:
        if base == "git":
            return _git_is_read_only(tokens)
        if base == "find":
            return _find_is_read_only(tokens)
        if base == "sed":
            return _sed_is_read_only(tokens)
        if base == "sort":
            return _sort_is_read_only(tokens)

    return True


def classify(command: str, strict: bool = True) -> Decision:
    """Return AUTO if every pipeline segment is provably read-only; else CONFIRM.

    Fail-closed: unknown, unparseable, or ambiguous input always returns CONFIRM.
    """
    if not command or not command.strip():
        return Decision.CONFIRM

    segments = _SEGMENT_SPLIT.split(command)

    for segment in segments:
        if not _segment_is_auto(segment, strict):
            return Decision.CONFIRM

    return Decision.AUTO
