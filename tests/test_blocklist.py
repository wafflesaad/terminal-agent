"""Tests for termagent.tools.blocklist: interactive-program refusal."""

from __future__ import annotations

import pytest

from termagent.tools.blocklist import check_blocked


# ---------------------------------------------------------------------------
# Direct blocked programs
# ---------------------------------------------------------------------------


def test_less_is_blocked():
    assert check_blocked("less foo") == "less"


def test_vim_is_blocked():
    assert check_blocked("vim file.py") == "vim"


def test_vi_is_blocked():
    assert check_blocked("vi /etc/hosts") == "vi"


def test_python_bare_is_blocked():
    assert check_blocked("python") == "python"


def test_python3_bare_is_blocked():
    assert check_blocked("python3") == "python3"


def test_node_bare_is_blocked():
    assert check_blocked("node") == "node"


def test_ssh_is_blocked():
    assert check_blocked("ssh user@host") == "ssh"


def test_top_is_blocked():
    assert check_blocked("top") == "top"


def test_tail_is_blocked():
    assert check_blocked("tail -f /var/log/syslog") == "tail"


# ---------------------------------------------------------------------------
# Safe commands return None
# ---------------------------------------------------------------------------


def test_grep_is_safe():
    assert check_blocked("grep x foo") is None


def test_ls_is_safe():
    assert check_blocked("ls -la") is None


def test_echo_is_safe():
    assert check_blocked("echo hello") is None


def test_cat_is_safe():
    assert check_blocked("cat file.txt") is None


def test_cd_is_safe():
    assert check_blocked("cd /tmp") is None


def test_empty_string_is_safe():
    assert check_blocked("") is None


# ---------------------------------------------------------------------------
# Pipeline and compound commands
# ---------------------------------------------------------------------------


def test_blocked_in_pipeline():
    assert check_blocked("cat a | vim -") == "vim"


def test_blocked_after_and():
    assert check_blocked("ls && less file") == "less"


def test_blocked_after_semicolon():
    assert check_blocked("echo hi; python") == "python"


def test_blocked_after_or():
    assert check_blocked("false || top") == "top"


def test_safe_pipeline_returns_none():
    assert check_blocked("cat file | grep pattern | wc -l") is None


def test_first_blocked_program_returned():
    # vim appears before less; either is acceptable but vim is first segment
    result = check_blocked("cat a | vim - | less")
    assert result == "vim"


# ---------------------------------------------------------------------------
# Path-prefixed commands: basename is checked
# ---------------------------------------------------------------------------


def test_full_path_blocked():
    assert check_blocked("/usr/bin/vim file.txt") == "vim"


def test_full_path_safe():
    assert check_blocked("/usr/bin/grep pattern file") is None
