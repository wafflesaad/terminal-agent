Run a security audit and code review in parallel on the target files.

## Instructions

Use the Task tool to launch both subagents simultaneously — do NOT wait for one to finish before starting the other.

**Task 1 — Security Audit:**
Invoke the `security-auditor` agent on $ARGUMENTS (or the full codebase if no argument given). Ask it to audit for all vulnerability classes it covers.

**Task 2 — Code Review:**
Invoke the `code-reviewer` agent on $ARGUMENTS (or the full codebase if no argument given). Ask it to review for quality, bugs, and best practices.

## After both complete

Combine the results into a single report with two sections:
1. 🔒 Security Audit Results
2. 🧹 Code Review Results

At the end, add a **Priority Action List** — the top 5 issues across both reports ranked by severity and effort.