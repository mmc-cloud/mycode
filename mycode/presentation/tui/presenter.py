"""Translate application and agent events into TUI view updates."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Protocol

from mycode.agent.events import AgentEvent
from mycode.application.events import RuntimeEvent
from mycode.event_format import summarize_event_content, summarize_tool_arguments

SUBAGENT_TUI_SUMMARY_CHARS = 160


class TuiConversation(Protocol):
    def add_user_message(self, content: str) -> None: ...

    def append_assistant_delta(self, content: str) -> None: ...

    def add_notice(self, content: str, *, level: str = "info") -> None: ...

    def add_tool_activity(self, name: str, arguments: str) -> int: ...

    def complete_tool_activity(
        self,
        token: int,
        name: str,
        *,
        ok: bool,
        summary: str,
    ) -> None: ...


class TuiHeader(Protocol):
    def set_session(self, value: str) -> None: ...


class TuiStatus(Protocol):
    def set_status(self, value: str) -> None: ...


@dataclass
class TuiPresenter:
    """Keep event interpretation out of Textual widgets.

    The presenter accepts the existing RuntimeEvent and AgentEvent contracts.
    It deliberately has no dependency on AgentRunner or AgentApplicationSession;
    the TUI application owns startup and feeds the resulting events here.
    """

    conversation: TuiConversation
    header: TuiHeader
    status: TuiStatus
    _pending_tool_rows: deque[tuple[int, str]] = field(
        default_factory=deque,
        init=False,
    )

    def show_user_message(self, content: str) -> None:
        self.conversation.add_user_message(content)
        self.status.set_status("UI Ready")

    def present(self, event: RuntimeEvent) -> None:
        if event.type == "runtime_ready":
            self._present_runtime_ready(event)
            return
        if event.type == "mcp_status":
            self._present_mcp_status(event)
            return
        if event.type == "agent" and event.agent_event is not None:
            self.present_agent_event(event.agent_event)
            return
        if event.type == "turn_finished":
            self._pending_tool_rows.clear()
            self.status.set_status("Ready")

    def present_agent_event(self, event: AgentEvent) -> None:
        if event.type in {"reasoning_delta", "reasoning_state"}:
            return

        if event.type == "turn":
            if event.turn_number is not None and event.max_turns is not None:
                self.conversation.add_notice(
                    f"· turn {event.turn_number}/{event.max_turns}"
                )
            if event.content:
                self.conversation.add_notice(
                    f"· turn notice {summarize_event_content(event.content)}"
                )
            return

        if event.type == "context":
            summary = summarize_event_content(event.content)
            if summary:
                self.conversation.add_notice(f"· context {summary}")
            return

        if event.type == "progress":
            if event.progress is None:
                return
            progress = event.progress
            self.conversation.add_notice(
                "· progress "
                f"reason={progress.reason} "
                f"stagnation={progress.stagnation_turns} "
                f"repeat={progress.same_tool_repeat} "
                f"result_repeat={progress.same_result_repeat} "
                f"resource_repeat={progress.resource_repeat}"
            )
            return

        if event.type == "model_start":
            self.status.set_status("Thinking…")
            return

        if event.type == "model_retry":
            if event.model_retry is None:
                return
            retry = event.model_retry
            self.status.set_status("Retrying…")
            self.conversation.add_notice(
                "⚠ model retry "
                f"{retry.attempt}/{retry.max_retries} "
                f"after {retry.delay_seconds:.1f}s "
                f"error={retry.error_type}",
                level="warning",
            )
            return

        if event.type == "text_delta":
            if event.content:
                self.status.set_status("Responding…")
                self.conversation.append_assistant_delta(event.content)
            return

        if event.type == "tool_call":
            if event.tool_call is None:
                return
            tool_call = event.tool_call
            arguments = summarize_tool_arguments(
                tool_call.name,
                tool_call.arguments,
            )
            token = self.conversation.add_tool_activity(
                tool_call.name,
                arguments,
            )
            self._pending_tool_rows.append((token, tool_call.name))
            self.status.set_status("Running…")
            return

        if event.type == "tool_result":
            if event.tool_result is None:
                return
            result = event.tool_result
            if self._pending_tool_rows:
                token, name = self._pending_tool_rows.popleft()
            else:
                metadata_name = result.metadata.get("tool_name")
                name = metadata_name if isinstance(metadata_name, str) else "tool"
                token = self.conversation.add_tool_activity(name, "")
            summary = summarize_event_content(
                result.content if result.ok else result.error
            )
            self.conversation.complete_tool_activity(
                token,
                name,
                ok=result.ok,
                summary=summary,
            )
            self.status.set_status("Running…" if not result.ok else "Thinking…")
            return

        if event.type == "artifact_warning":
            summary = summarize_event_content(event.content)
            if summary:
                self.conversation.add_notice(f"⚠ {summary}", level="warning")
            return

        if event.type == "error":
            summary = summarize_event_content(event.error or event.content)
            self.conversation.add_notice(f"✗ {summary}", level="error")
            self.status.set_status("Error")
            return

        if event.type == "stop":
            reason = event.stop_reason or "unknown"
            content = summarize_event_content(event.content)
            message = f"· stop reason={reason}"
            if content:
                message += f" {content}"
            self.conversation.add_notice(message)
            self.status.set_status("Stopped")

    def present_subagent_transition(self, transition) -> None:
        """Render the few high-value SubAgent state transitions.

        Completed and failed envelopes are shown once from on_result; the
        intermediate transitions only annotate start, waiting for permission
        and interruption (which may never reach on_result on an interrupt).
        """
        role = transition.role.capitalize()
        if transition.state == "running" and transition.reason == "run_started":
            self.conversation.add_notice(f"› {role} started")
            return
        if transition.state == "awaiting_confirmation":
            self.conversation.add_notice(f"· {role} waiting for permission")
            return
        if transition.state == "interrupted":
            self.conversation.add_notice(
                f"⚠ {role} interrupted",
                level="warning",
            )

    def present_subagent_result(self, execution) -> None:
        result = execution.result
        role = result.role.capitalize()
        if result.status == "completed":
            marker, level = "✓", "info"
        elif result.status == "interrupted":
            marker, level = "⚠", "warning"
        else:
            marker, level = "✗", "error"
        line = f"{marker} {role} {result.status}"
        summary = _subagent_summary(result.summary)
        if summary:
            line += f" {summary}"
        self.conversation.add_notice(line, level=level)

    def _present_runtime_ready(self, event: RuntimeEvent) -> None:
        session = event.session_title or event.session_id or "—"
        self.header.set_session(session)
        self.conversation.add_notice(f"· runtime ready session={session}")
        for warning in (*event.instruction_warnings, *event.skill_warnings):
            summary = summarize_event_content(warning)
            if summary:
                self.conversation.add_notice(
                    f"⚠ startup {summary}",
                    level="warning",
                )
        self.status.set_status("UI Ready")

    def _present_mcp_status(self, event: RuntimeEvent) -> None:
        if event.mcp_status is None:
            return
        status = event.mcp_status
        if status.status == "connected":
            self.conversation.add_notice(
                f"· MCP {status.alias} connected ({status.tool_count} tools)"
            )
            return
        detail = summarize_event_content(
            status.error_summary or status.error_type or "unavailable"
        )
        self.conversation.add_notice(
            f"⚠ MCP {status.alias} unavailable: {detail}",
            level="warning",
        )


def _subagent_summary(value: str) -> str:
    normalized = " ".join(value.split())
    if len(normalized) <= SUBAGENT_TUI_SUMMARY_CHARS:
        return normalized
    return normalized[: SUBAGENT_TUI_SUMMARY_CHARS - 3].rstrip() + "..."
