"""Tests for termagent.tools.classify: allowlist-based, fail-closed command classifier."""

from __future__ import annotations

import pytest

from termagent.tools.classify import Decision, classify


# ---------------------------------------------------------------------------
# Basic read-only commands → AUTO
# ---------------------------------------------------------------------------


def test_ls_long():
    assert classify("ls -la") == Decision.AUTO


def test_cat_file():
    assert classify("cat foo.txt") == Decision.AUTO


def test_pwd():
    assert classify("pwd") == Decision.AUTO


def test_echo_bare():
    assert classify("echo hello") == Decision.AUTO


def test_grep():
    assert classify("grep pattern file.txt") == Decision.AUTO


def test_wc():
    assert classify("wc -l file.txt") == Decision.AUTO


def test_diff():
    assert classify("diff a.txt b.txt") == Decision.AUTO


# ---------------------------------------------------------------------------
# Mutating/unknown commands → CONFIRM
# ---------------------------------------------------------------------------


def test_rm_rf():
    assert classify("rm -rf build") == Decision.CONFIRM


def test_mv():
    assert classify("mv a b") == Decision.CONFIRM


def test_cp():
    assert classify("cp a b") == Decision.CONFIRM


def test_unknown_binary():
    assert classify("some-unknown-binary") == Decision.CONFIRM


def test_touch():
    assert classify("touch file.txt") == Decision.CONFIRM


def test_mkdir():
    assert classify("mkdir newdir") == Decision.CONFIRM


# ---------------------------------------------------------------------------
# Redirection → CONFIRM
# ---------------------------------------------------------------------------


def test_output_redirect():
    assert classify("echo hi > out.txt") == Decision.CONFIRM


def test_append_redirect():
    assert classify("echo hi >> out.txt") == Decision.CONFIRM


def test_stderr_redirect():
    assert classify("ls 2> err.txt") == Decision.CONFIRM


def test_input_redirect_is_auto():
    # < alone is read-only (input redirection does not write)
    assert classify("cat < in.txt") == Decision.AUTO


# ---------------------------------------------------------------------------
# Pipelines
# ---------------------------------------------------------------------------


def test_pipe_both_safe():
    assert classify("cat a | grep x") == Decision.AUTO


def test_pipe_tee_writes():
    assert classify("cat a | tee b") == Decision.CONFIRM


def test_pipe_long_safe():
    assert classify("cat a | grep x | sort | uniq") == Decision.AUTO


def test_pipe_one_bad_segment():
    assert classify("ls; rm x") == Decision.CONFIRM


def test_pipe_and_bad():
    assert classify("ls && rm x") == Decision.CONFIRM


def test_pipe_or_bad():
    assert classify("ls || rm x") == Decision.CONFIRM


def test_newline_separator_safe():
    assert classify("ls\ncat f") == Decision.AUTO


def test_newline_separator_bad():
    assert classify("ls\nrm f") == Decision.CONFIRM


def test_all_safe_chain():
    assert classify("cat a | grep x | sort | uniq | wc -l") == Decision.AUTO


# ---------------------------------------------------------------------------
# Git sub-commands
# ---------------------------------------------------------------------------


def test_git_status():
    assert classify("git status") == Decision.AUTO


def test_git_log():
    assert classify("git log --oneline") == Decision.AUTO


def test_git_diff():
    assert classify("git diff HEAD") == Decision.AUTO


def test_git_show():
    assert classify("git show abc123") == Decision.AUTO


def test_git_branch():
    assert classify("git branch") == Decision.AUTO


def test_git_remote_v():
    assert classify("git remote -v") == Decision.AUTO


def test_git_commit():
    assert classify("git commit -m x") == Decision.CONFIRM


def test_git_push():
    assert classify("git push") == Decision.CONFIRM


def test_git_checkout():
    assert classify("git checkout main") == Decision.CONFIRM


def test_git_reset():
    assert classify("git reset --hard HEAD") == Decision.CONFIRM


def test_git_clean():
    assert classify("git clean -fd") == Decision.CONFIRM


def test_git_merge():
    assert classify("git merge main") == Decision.CONFIRM


def test_git_rebase():
    assert classify("git rebase main") == Decision.CONFIRM


def test_git_add():
    assert classify("git add .") == Decision.CONFIRM


def test_git_bare():
    assert classify("git") == Decision.CONFIRM


# ---------------------------------------------------------------------------
# find sub-checks
# ---------------------------------------------------------------------------


def test_find_name():
    assert classify("find . -name '*.py'") == Decision.AUTO


def test_find_type():
    assert classify("find . -type f") == Decision.AUTO


def test_find_delete():
    assert classify("find . -delete") == Decision.CONFIRM


def test_find_exec():
    assert classify("find . -exec rm {} \\;") == Decision.CONFIRM


def test_find_execdir():
    assert classify("find . -execdir rm {} \\;") == Decision.CONFIRM


def test_find_ok():
    assert classify("find . -name x -ok rm {} \\;") == Decision.CONFIRM


def test_find_okdir():
    assert classify("find . -okdir rm {} \\;") == Decision.CONFIRM


# ---------------------------------------------------------------------------
# sed sub-checks
# ---------------------------------------------------------------------------


def test_sed_plain():
    assert classify("sed 's/a/b/' f") == Decision.AUTO


def test_sed_inplace():
    assert classify("sed -i 's/a/b/' f") == Decision.CONFIRM


def test_sed_inplace_backup():
    assert classify("sed -i.bak 's/a/b/' f") == Decision.CONFIRM


def test_sed_inplace_long():
    assert classify("sed --in-place 's/a/b/' f") == Decision.CONFIRM


# ---------------------------------------------------------------------------
# sort sub-checks
# ---------------------------------------------------------------------------


def test_sort_plain():
    assert classify("sort f") == Decision.AUTO


def test_sort_output_flag():
    assert classify("sort -o out f") == Decision.CONFIRM


def test_sort_output_long():
    assert classify("sort --output=out f") == Decision.CONFIRM


# ---------------------------------------------------------------------------
# sudo / privilege escalation
# ---------------------------------------------------------------------------


def test_sudo_ls():
    assert classify("sudo ls") == Decision.CONFIRM


def test_sudo_cat():
    assert classify("sudo cat /etc/shadow") == Decision.CONFIRM


def test_doas_ls():
    assert classify("doas ls") == Decision.CONFIRM


# ---------------------------------------------------------------------------
# Env-variable prefix stripping
# ---------------------------------------------------------------------------


def test_env_prefix_safe():
    assert classify("FOO=bar ls") == Decision.AUTO


def test_env_prefix_unsafe():
    assert classify("FOO=bar rm x") == Decision.CONFIRM


def test_multiple_env_prefix_safe():
    assert classify("FOO=bar BAZ=qux cat f") == Decision.AUTO


# ---------------------------------------------------------------------------
# Path-prefixed commands
# ---------------------------------------------------------------------------


def test_full_path_git_status():
    assert classify("/usr/bin/git status") == Decision.AUTO


def test_full_path_rm():
    assert classify("/bin/rm x") == Decision.CONFIRM


def test_full_path_ls():
    assert classify("/usr/bin/ls -la") == Decision.AUTO


# ---------------------------------------------------------------------------
# Fail-closed edge cases
# ---------------------------------------------------------------------------


def test_empty_string():
    assert classify("") == Decision.CONFIRM


def test_whitespace_only():
    assert classify("   ") == Decision.CONFIRM


def test_unparseable_quotes():
    assert classify('echo "unterminated') == Decision.CONFIRM


# ---------------------------------------------------------------------------
# strict knob
# ---------------------------------------------------------------------------


def test_strict_true_git_status():
    assert classify("git status", strict=True) == Decision.AUTO


def test_strict_false_git_status():
    assert classify("git status", strict=False) == Decision.AUTO


def test_strict_true_git_commit():
    assert classify("git commit -m x", strict=True) == Decision.CONFIRM


def test_strict_false_unknown():
    # Fail-closed invariant: unknown → CONFIRM regardless of strict
    assert classify("some-unknown-binary", strict=False) == Decision.CONFIRM


def test_strict_false_rm():
    assert classify("rm -rf build", strict=False) == Decision.CONFIRM
