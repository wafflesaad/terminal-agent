# 05 — Command classifier

## Goal

Decide, for a given shell command, whether it may run automatically or must be confirmed by the
user. Allowlist-based and **fail-closed**: known read-only → auto; everything else → confirm.
This is the safety core. It is a pure function and must be the most heavily tested unit in the
codebase.

## Depends on

- 01 (`strict_mode`).

## Deliverables

- `termagent/tools/classify.py`.
- `tests/test_classify.py` (extensive).

## Design

```python
class Decision(Enum):
    AUTO = "auto"        # read-only, run without asking
    CONFIRM = "confirm"  # requires user y/n

def classify(command: str, strict: bool = True) -> Decision: ...
```

Rules, in order. The default for anything not positively recognized as read-only is `CONFIRM`.

1. **Split into pipeline segments** on `|`, `;`, `&&`, `||`, and newlines. A command is `AUTO`
   only if **every** segment is `AUTO`. Any segment in doubt → the whole thing is `CONFIRM`.
2. **Redirections force confirm.** If a segment contains `>`, `>>`, or a process-substitution
   write, it writes to disk → `CONFIRM`. (`<` input redirection alone is read-only.)
3. **`sudo` / privilege escalation → always `CONFIRM`** (or block — your call; confirm is fine).
4. **Per-segment check:** parse with `shlex.split`; take the first token (strip leading env
   assignments like `FOO=bar cmd`). If the base command is in `READ_ONLY` → that segment is
   `AUTO`. Otherwise → `CONFIRM`.
5. **Unknown command → `CONFIRM`.** Never default unknown to auto.

Subtlety to guard against (write a test for each): a read-only base command with a writing flag.
The cheapest safe policy is to treat a small set of "mostly-read" commands as `AUTO` only in
their plain form and `CONFIRM` if they carry known mutating flags:

- `git`: `AUTO` for `status|log|diff|show|branch|remote -v`; `CONFIRM` for everything else
  (`commit|push|checkout|reset|clean|rm|merge|rebase`...).
- `find`: `AUTO` unless it contains `-delete` or `-exec`/`-ok` (those run arbitrary commands).
- `sort`, `sed`, `awk`: `AUTO` only without in-place flags (`sed -i`, etc.); else `CONFIRM`.

Keep the allowlist conservative. It is fine — preferred — to over-confirm early.

### Starter allowlist

```python
READ_ONLY = {
    "ls","pwd","cat","head","tail","wc","stat","file","tree","du","df","find",
    "grep","rg","egrep","fgrep","which","whereis","type","echo","printf",
    "ps","env","printenv","date","whoami","id","uname","hostname","uptime",
    "git","sort","uniq","cut","tr","diff","cmp","sed","awk","jq","column","basename","dirname",
}
```

(`top`/`htop` etc. are handled by the spec 04 blocklist, separately from classification. The
classifier answers "auto vs confirm"; the blocklist answers "can this run at all".)

`strict` is a tightening knob: when `strict=True` (default), the `git`/`find`/`sed` flag checks
above are enforced; you may relax some of them when `strict=False`. The fail-closed default
never changes regardless of `strict`.

## Acceptance criteria

Tests must include at least:

- `classify("ls -la")` → `AUTO`; `classify("cat foo.txt")` → `AUTO`.
- `classify("rm -rf build")` → `CONFIRM`; `classify("mv a b")` → `CONFIRM`.
- `classify("echo hi > out.txt")` → `CONFIRM` (redirection).
- `classify("cat a | grep x")` → `AUTO`; `classify("cat a | tee b")` → `CONFIRM` (tee writes).
- `classify("git status")` → `AUTO`; `classify("git commit -m x")` → `CONFIRM`;
  `classify("git push")` → `CONFIRM`.
- `classify("find . -name '*.py'")` → `AUTO`; `classify("find . -delete")` → `CONFIRM`.
- `classify("sed 's/a/b/' f")` → `AUTO`; `classify("sed -i 's/a/b/' f")` → `CONFIRM`.
- `classify("sudo ls")` → `CONFIRM`.
- `classify("some-unknown-binary")` → `CONFIRM` (fail-closed).
- `classify("ls; rm x")` → `CONFIRM` (one bad segment taints the whole).

## Out of scope / deferred

- A learned/dynamic allowlist or per-project overrides. Static set for now.
