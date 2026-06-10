# 04 — Shell executor

## Goal

The single tool the agent gets: run a shell command safely and return structured output.
Owns working-directory tracking, the `cd` special-case, timeouts, output capping, and refusal
of interactive programs. No agent logic here — this is a pure mechanics layer, easy to test.

## Depends on

- 01 (`timeout_seconds`, `max_output_chars`).

## Deliverables

- `termagent/tools/shell.py` — `run_command`, `apply_cd`, output capping.
- `termagent/tools/blocklist.py` — interactive-program refusal.
- `tests/test_shell.py`, `tests/test_blocklist.py`.

## Design

### Result type

```python
@dataclass
class ExecResult:
    stdout: str
    stderr: str
    exit_code: int
    duration_s: float
    timed_out: bool
    truncated: bool
```

### Working directory

The agent's cwd lives in graph state (spec 06). Each command runs with `cwd=` passed to
`subprocess.run`. A fresh subprocess does not persist `cd`, so we special-case it:

```python
def apply_cd(command: str, cwd: str) -> str | None:
    # If command is a bare `cd <path>` (or `cd` alone → home), resolve <path> relative to cwd,
    # verify it exists and is a directory, and RETURN the new absolute cwd.
    # Return None if the command is not a pure cd (then it runs normally).
    # Raise a clear error if the target does not exist / is not a dir.
```

The execute node (spec 06) calls `apply_cd` first; if it returns a path, update state cwd and
skip subprocess entirely. Only handle the simple `cd <path>` form — compound commands like
`cd x && make` still run as a subprocess (and `cd` inside them is naturally scoped to that
subprocess, which is fine).

### Execution

```python
def run_command(command: str, cwd: str, timeout_s: int, max_output_chars: int) -> ExecResult:
    # subprocess.run(command, shell=True, cwd=cwd, capture_output=True, text=True,
    #                timeout=timeout_s, start_new_session=True)
    # shell=True is required for pipes/redirects; the confirmation gate (spec 05/06) is what
    # makes this acceptable.
```

- `start_new_session=True` puts the child in its own process group. On `TimeoutExpired`, kill
  the whole group (`os.killpg(os.getpgid(pid), SIGKILL)`) so pipelines don't leave orphans, and
  return `timed_out=True` with whatever partial output was captured.
- Capture exit code and stderr; never raise on a non-zero exit — that's normal data for the
  agent to react to.

### Output capping

```python
def cap_output(text: str, max_chars: int) -> tuple[str, bool]:
    # If len(text) <= max_chars: return (text, False).
    # Else keep a head and a tail and drop the middle, inserting a marker like
    #   "\n…[truncated N chars]…\n". Return (capped, True).
```

Cap stdout and stderr independently before they go back into context.

### Interactive / TTY blocklist

```python
BLOCKED = {"vim","vi","nano","emacs","less","more","top","htop","man","ssh","tmux",
           "screen","watch","tail",  # `tail -f` hangs; see note
           "python","python3","node","irb","psql","mysql","sqlite3"}  # bare REPLs hang

def check_blocked(command: str) -> str | None:
    # Parse the first token of each pipeline segment with shlex; if any is in BLOCKED
    # (and not given a non-interactive flag), return the offending program name; else None.
```

- The execute node refuses a blocked command up front with a message like:
  `"<prog> is interactive and can't run here. Try a non-interactive form (e.g. `grep` instead
  of `less`, `cat file` instead of opening it in an editor)."`
- This refusal becomes a `ToolMessage` so the agent re-plans, rather than hanging the loop.
- `tail` and bare interpreters are blocked because they wait for input/never exit; allow
  `tail` only with an explicit line count if you want to special-case it later (out of scope).

## Acceptance criteria

- `run_command` returns captured stdout/stderr/exit code for a normal command; a sleep beyond
  the timeout returns `timed_out=True` and leaves no orphan process (verify the group is gone).
- `apply_cd("cd ..", cwd)` returns the parent path; `apply_cd("cd /nope", cwd)` raises;
  `apply_cd("ls", cwd)` returns `None`.
- `cap_output` truncates over-budget text with a marker and reports `truncated=True`.
- `check_blocked("less foo")` → `"less"`; `check_blocked("grep x foo")` → `None`;
  `check_blocked("cat a | vim -")` → `"vim"`.
- All of the above are unit-testable without the agent or any model.

## Out of scope / deferred

- A persistent shell (pexpect) for env-var/`source` persistence. Not now.
- Streaming a command's output live (capture-then-return is fine for v1).
