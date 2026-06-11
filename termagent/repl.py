"""REPL: read a line, run the agent graph, stream output, handle approval interrupts."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Command
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel

from termagent import config
from termagent.context import Context
from termagent.router import dispatch

if TYPE_CHECKING:
    from langgraph.checkpoint.base import BaseCheckpointSaver
    from langgraph.graph.state import CompiledStateGraph

    from termagent.config import Settings
    from termagent.store.sessions import Session, SessionStore


class Repl:
    def __init__(
        self,
        settings: "Settings",
        store: "SessionStore",
        checkpointer: "BaseCheckpointSaver",
        graph: "CompiledStateGraph",
        session: "Session",
        *,
        console: Console | None = None,
        context: Context | None = None,
    ) -> None:
        if context is None:
            context = Context(
                settings=settings,
                store=store,
                checkpointer=checkpointer,
                graph=graph,
                session=session,
                provider_name=settings.default_provider,
                cwd=os.getcwd(),
                auto_approve=settings.auto_approve,
                console=console or Console(),
            )
        self._ctx = context
        self._console = self._ctx.console

    # Backwards-compatible read-only views onto the live context.
    @property
    def _settings(self) -> "Settings":
        return self._ctx.settings

    @property
    def _store(self) -> "SessionStore":
        return self._ctx.store

    @property
    def _graph(self) -> "CompiledStateGraph":
        return self._ctx.graph

    @property
    def _session(self) -> "Session":
        return self._ctx.session

    def run(self) -> None:
        """Start the interactive REPL loop."""
        from prompt_toolkit import PromptSession
        from prompt_toolkit.history import FileHistory

        history_path = config.data_dir() / "history"
        history_path.parent.mkdir(parents=True, exist_ok=True)

        prompt_session: PromptSession = PromptSession(
            history=FileHistory(str(history_path))
        )

        self._console.print(
            f"[bold]termagent[/bold]  "
            f"[{self._ctx.provider_name}]  {self._ctx.model_label()}  "
            f"session:{self._ctx.session.id}  "
            f"[dim]type /help for commands, /exit to quit[/dim]"
        )

        while True:
            try:
                line = prompt_session.prompt("» ")
            except EOFError:
                break
            except KeyboardInterrupt:
                continue

            line = line.strip()
            if not line:
                continue

            if line.startswith("/"):
                result = dispatch(self._ctx, line)
                if result.action == "exit":
                    break
                if result.action == "prompt" and result.prompt:
                    line = result.prompt
                else:
                    continue

            try:
                self.handle_task(line)
            except KeyboardInterrupt:
                self._console.print("\n[dim]^C cancelled[/dim]")
            except Exception as exc:
                self._console.print(f"[red]model error:[/red] {exc}")

    def handle_task(self, text: str) -> None:
        """Run the agent graph for one user task, handling approval interrupts."""
        self._ctx.last_prompt = text
        cfg: dict = self._ctx.thread_config()
        payload: dict = {"messages": [HumanMessage(content=text)], "cwd": self._ctx.cwd}
        rendered = 0

        with self._console.status("[dim]thinking…[/dim]"):
            for chunk in self._graph.stream(payload, config=cfg, stream_mode="values"):
                rendered = self._render(chunk, rendered)

        snap = self._graph.get_state(cfg)

        while snap.next:
            intr = snap.tasks[0].interrupts[0].value
            if self._ctx.auto_approve:
                self._console.print(
                    "[yellow]⚠ auto-approve ON — running without confirmation[/yellow]"
                )
                ans = True
            else:
                ans = self.confirm(intr["command"], intr["cwd"], intr["reason"])

            with self._console.status("[dim]thinking…[/dim]"):
                for chunk in self._graph.stream(
                    Command(resume=ans), config=cfg, stream_mode="values"
                ):
                    rendered = self._render(chunk, rendered)

            snap = self._graph.get_state(cfg)

        message_count = len(snap.values.get("messages", []))
        self._ctx.store.touch(self._ctx.session.id, message_count)
        if self._ctx.session.title == "":
            title = text[:60]
            self._ctx.store.set_title(self._ctx.session.id, title)
            self._ctx.session.title = title

        self._ctx.cwd = snap.values.get("cwd", self._ctx.cwd)

    def confirm(self, command: str, cwd: str, reason: str) -> bool:
        """Render the approval panel; return True only when the user answers y."""
        content = (
            f"[bold]Command:[/bold] {escape(command)}\n"
            f"[bold]Directory:[/bold] {escape(cwd)}\n"
            f"[bold]Reason:[/bold] {escape(reason)}"
        )
        self._console.print(Panel(content, title="Confirm", border_style="yellow"))
        answer = self._console.input("Run this? [y/N] ").strip().lower()
        return answer == "y"

    def _render(self, chunk: dict, rendered: int) -> int:
        """Render newly arrived messages from a values-mode stream chunk."""
        messages = chunk.get("messages", [])
        for msg in messages[rendered:]:
            if isinstance(msg, ToolMessage):
                self._console.print(f"[dim]⎿ {escape(msg.content)}[/dim]")
            elif isinstance(msg, AIMessage) and msg.tool_calls:
                if self._ctx.verbose:
                    self._console.print(f"[dim italic]{escape(msg.content)}[/dim italic]")
            elif isinstance(msg, AIMessage) and not msg.tool_calls and msg.content:
                self._console.print(escape(msg.content))
        return len(messages)
