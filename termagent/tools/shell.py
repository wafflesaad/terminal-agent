"""Shell executor: structured execution, cd handling, timeouts, and output capping."""

from __future__ import annotations

import os
import shlex
import signal
import subprocess
import time
from dataclasses import dataclass


@dataclass
class ExecResult:
    stdout: str
    stderr: str
    exit_code: int
    duration_s: float
    timed_out: bool
    truncated: bool


def cap_output(text: str, max_chars: int) -> tuple[str, bool]:
    """Return (text, False) if within budget; else (head+marker+tail, True)."""
    if len(text) <= max_chars:
        return (text, False)
    marker_template = "\n…[truncated {} chars]…\n"
    # Reserve chars for the marker (use a rough estimate first, then refine)
    marker_len = len(marker_template.format(len(text)))
    usable = max_chars - marker_len
    if usable < 0:
        usable = 0
    half = usable // 2
    head = text[:half]
    tail = text[len(text) - (usable - half):]
    dropped = len(text) - len(head) - len(tail)
    marker = marker_template.format(dropped)
    return (head + marker + tail, True)


def apply_cd(command: str, cwd: str) -> str | None:
    """Return the new absolute cwd if command is a bare `cd [path]`, else None.

    Raises FileNotFoundError or NotADirectoryError if the target does not exist
    or is not a directory. Compound commands (cd x && make) return None.
    """
    try:
        tokens = shlex.split(command)
    except ValueError:
        return None

    if not tokens or tokens[0] != "cd":
        return None

    # Compound: more than one argument to cd, or shell operators present
    # shlex.split strips operators only within quotes — a bare && would be two tokens
    # but we also want to reject "cd x && make" style. If the token list has >2 items
    # after the cd, it's compound → let subprocess handle it.
    if len(tokens) > 2:
        return None

    if len(tokens) == 1 or tokens[1] in ("~", ""):
        target = os.path.expanduser("~")
    else:
        target = os.path.expanduser(tokens[1])
        if not os.path.isabs(target):
            target = os.path.normpath(os.path.join(cwd, target))

    if not os.path.exists(target):
        raise FileNotFoundError(f"cd: no such file or directory: {target}")
    if not os.path.isdir(target):
        raise NotADirectoryError(f"cd: not a directory: {target}")

    return os.path.abspath(target)


def run_command(
    command: str,
    cwd: str,
    timeout_s: int,
    max_output_chars: int,
) -> ExecResult:
    """Run *command* in a shell, bounded by timeout_s; return structured output.

    shell=True is required for pipes/redirects; safety comes from the confirmation
    gate in spec 05/06, not from sanitising the command string here.
    """
    kwargs: dict = dict(
        shell=True,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    # POSIX: own process group so killpg cleans up the whole pipeline on timeout.
    if os.name != "nt":
        kwargs["start_new_session"] = True

    start = time.monotonic()
    proc = subprocess.Popen(command, **kwargs)  # noqa: S602 — intentional shell=True
    timed_out = False

    try:
        stdout_raw, stderr_raw = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        if os.name != "nt":
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
        else:
            proc.kill()
        stdout_raw, stderr_raw = proc.communicate()

    duration_s = time.monotonic() - start
    exit_code = proc.returncode if proc.returncode is not None else -1

    stdout, trunc_out = cap_output(stdout_raw or "", max_output_chars)
    stderr, trunc_err = cap_output(stderr_raw or "", max_output_chars)

    return ExecResult(
        stdout=stdout,
        stderr=stderr,
        exit_code=exit_code,
        duration_s=duration_s,
        timed_out=timed_out,
        truncated=trunc_out or trunc_err,
    )
