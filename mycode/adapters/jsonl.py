"""Versioned JSONL adapter for the application runtime."""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO
from uuid import uuid4

from mycode.agent.events import (
    AgentEvent,
    AgentModelRetry,
    AgentProgressSnapshot,
    AgentToolCall,
)
from mycode.application.agent_session import AgentApplicationSession, start_agent_application_session
from mycode.application.events import RuntimeEvent
from mycode.application.sessions import SessionStartRequest
from mycode.config import LLMConfig
from mycode.mcp import (
    MCPConfig,
    MCPConfigError,
    MCPServerStatus,
    load_mcp_config_layers,
    resolve_project_mcp_trust,
)
from mycode.mcp.trust import MCPTrustRequest, MCPTrustServer, MCPTrustWarning
from mycode.permissions import (
    ConfirmationRequest,
    ConfirmationResult,
    Confirmer,
)
from mycode.persistence.session_store import (
    SessionInUseError,
    SessionNotFoundError,
    SessionStore,
    SessionStoreError,
)
from mycode.project import ProjectIdentity
from mycode.tools import ToolResult
from mycode.tools.workspace import Workspace


JSONL_PROTOCOL_VERSION = 1
_PERMISSION_DECISIONS = {"once", "task", "session", "reject"}


class JsonlProtocolError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass
class JsonlChannel:
    input_stream: TextIO
    output_stream: TextIO
    error_stream: TextIO | None = None

    def emit(self, message: Mapping[str, object]) -> None:
        if not isinstance(message, Mapping):
            raise TypeError("JSONL output must be an object")
        payload = {"version": JSONL_PROTOCOL_VERSION, **dict(message)}
        if payload.get("version") != JSONL_PROTOCOL_VERSION or isinstance(
            payload.get("version"), bool
        ):
            raise ValueError("JSONL output version must be 1")
        if not isinstance(payload.get("type"), str):
            raise ValueError("JSONL output type must be a string")
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        )
        self.output_stream.write(encoded + "\n")
        self.output_stream.flush()

    def read_message(self) -> dict[str, object] | None:
        line = self.input_stream.readline()
        if line == "":
            return None
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as error:
            raise JsonlProtocolError("invalid_json", "Input is not valid JSON.") from error
        if not isinstance(payload, dict):
            raise JsonlProtocolError("invalid_message", "Input message must be a JSON object.")
        if payload.get("version") != JSONL_PROTOCOL_VERSION or isinstance(
            payload.get("version"), bool
        ):
            raise JsonlProtocolError("unsupported_version", "Unsupported JSONL protocol version.")
        if not isinstance(payload.get("type"), str):
            raise JsonlProtocolError("missing_type", "Input message requires a string type.")
        return payload

    def diagnostic(self, message: str) -> None:
        if self.error_stream is not None:
            self.error_stream.write(message + "\n")
            self.error_stream.flush()


class JsonlConfirmer(Confirmer):
    def __init__(self, channel: JsonlChannel) -> None:
        self.channel = channel

    def confirm(self, request: ConfirmationRequest) -> ConfirmationResult:
        request_id = uuid4().hex
        permission_request = request.permission_request
        self.channel.emit(
            {
                "type": "permission_request",
                "request_id": request_id,
                "tool_name": permission_request.tool_name,
                "capability": permission_request.capability,
                "action": permission_request.action,
                "target": permission_request.target,
                "reason": request.permission_decision.reason,
                "prompt": request.prompt,
                "arguments": _json_safe(permission_request.arguments),
                "metadata": _json_safe(request.metadata),
            }
        )
        while True:
            try:
                message = self.channel.read_message()
            except JsonlProtocolError as error:
                self._protocol_error(error.code, str(error))
                continue
            if message is None:
                return ConfirmationResult.rejected(
                    message="Permission confirmation unavailable.",
                    metadata={"input": "eof"},
                )
            if message.get("type") != "permission_response":
                self._protocol_error(
                    "unexpected_message",
                    "Expected permission_response.",
                )
                continue
            if message.get("request_id") != request_id:
                self._protocol_error(
                    "request_id_mismatch",
                    "Permission response request_id does not match.",
                )
                continue
            decision = message.get("decision")
            if not isinstance(decision, str) or decision not in _PERMISSION_DECISIONS:
                self._protocol_error(
                    "invalid_permission_decision",
                    "Permission decision must be once, task, session, or reject.",
                )
                continue
            if decision == "reject":
                return ConfirmationResult.rejected(
                    message="Permission confirmation rejected.",
                    metadata={"input": decision},
                )
            return ConfirmationResult.approved(
                scope=decision,
                message="Permission confirmation approved.",
                metadata={"input": decision},
            )

    def _protocol_error(self, code: str, message: str) -> None:
        self.channel.emit(
            {
                "type": "runtime_error",
                "code": code,
                "message": message,
            }
        )


class JsonlMCPTrustConfirmer:
    def __init__(self, channel: JsonlChannel) -> None:
        self.channel = channel

    def confirm(self, request: MCPTrustRequest) -> bool:
        request_id = uuid4().hex
        self.channel.emit(
            {
                "type": "mcp_trust_request",
                "request_id": request_id,
                "servers": [_serialize_mcp_trust_server(server) for server in request.servers],
            }
        )
        while True:
            try:
                message = self.channel.read_message()
            except JsonlProtocolError as error:
                self._protocol_error(error.code, str(error))
                continue
            if message is None:
                return False
            if message.get("type") != "mcp_trust_response":
                self._protocol_error(
                    "unexpected_message",
                    "Expected mcp_trust_response.",
                )
                continue
            if message.get("request_id") != request_id:
                self._protocol_error(
                    "request_id_mismatch",
                    "MCP trust response request_id does not match.",
                )
                continue
            approved = message.get("approved")
            if not isinstance(approved, bool):
                self._protocol_error(
                    "invalid_mcp_trust_response",
                    "MCP trust approved must be a boolean.",
                )
                continue
            return approved

    def report_warning(self, warning: MCPTrustWarning) -> None:
        self.channel.emit(
            {
                "type": "runtime_warning",
                "code": warning.code,
                "message": warning.message,
            }
        )

    def _protocol_error(self, code: str, message: str) -> None:
        self.channel.emit(
            {
                "type": "runtime_error",
                "code": code,
                "message": message,
            }
        )


def run_jsonl_runtime(
    *,
    workspace_path: Path | None = None,
    session_request: SessionStartRequest | None = None,
    session_store: SessionStore | None = None,
    llm_config: LLMConfig | None = None,
    mcp_config: MCPConfig | None = None,
    input_stream: TextIO | None = None,
    output_stream: TextIO | None = None,
    error_stream: TextIO | None = None,
) -> int:
    channel = JsonlChannel(
        input_stream=sys.stdin if input_stream is None else input_stream,
        output_stream=sys.stdout if output_stream is None else output_stream,
        error_stream=sys.stderr if error_stream is None else error_stream,
    )
    workspace = Workspace(Path.cwd() if workspace_path is None else workspace_path)
    project = ProjectIdentity.from_workspace(workspace.root)
    store = SessionStore() if session_store is None else session_store
    request = (
        SessionStartRequest(mode="continue")
        if session_request is None
        else session_request
    )

    effective_mcp_config = mcp_config
    if effective_mcp_config is None:
        effective_mcp_config = _resolve_machine_mcp_config(channel, workspace, project)

    confirmer = JsonlConfirmer(channel)
    try:
        application_session = start_agent_application_session(
            store,
            project,
            request=request,
            mcp_config=effective_mcp_config,
            confirmer=confirmer,
            llm_config=llm_config,
        )
    except (SessionNotFoundError, SessionInUseError, SessionStoreError) as error:
        channel.emit(
            {
                "type": "runtime_error",
                "code": "session_start_failed",
                "message": str(error),
            }
        )
        return 1
    except Exception as error:  # noqa: BLE001 - adapter startup boundary
        channel.diagnostic(f"runtime startup failed: {type(error).__name__}: {error}")
        channel.emit(
            {
                "type": "runtime_error",
                "code": "startup_failed",
                "message": "Runtime startup failed.",
            }
        )
        return 1

    try:
        for event in application_session.startup_events():
            channel.emit(serialize_runtime_event(event))
        return _run_message_loop(channel, application_session)
    except KeyboardInterrupt:
        _interrupt_application_session(channel, application_session)
        return 130
    except Exception as error:  # noqa: BLE001 - runtime boundary
        channel.diagnostic(f"runtime failed: {type(error).__name__}: {error}")
        _interrupt_application_session(channel, application_session)
        return 1


def _resolve_machine_mcp_config(
    channel: JsonlChannel,
    workspace: Workspace,
    project: ProjectIdentity,
) -> MCPConfig:
    try:
        loaded = load_mcp_config_layers(workspace_root=workspace.root)
    except MCPConfigError as error:
        channel.emit(
            {
                "type": "runtime_error",
                "code": "mcp_config_error",
                "message": str(error),
            }
        )
        return MCPConfig()

    try:
        return resolve_project_mcp_trust(
            loaded,
            project,
            confirmer=JsonlMCPTrustConfirmer(channel),
        ).config
    except Exception as error:  # noqa: BLE001 - trust startup boundary
        channel.diagnostic(f"mcp trust resolution failed: {type(error).__name__}: {error}")
        channel.emit(
            {
                "type": "runtime_error",
                "code": "mcp_trust_failed",
                "message": "MCP trust resolution failed.",
            }
        )
        return MCPConfig()


def _run_message_loop(
    channel: JsonlChannel,
    application_session: AgentApplicationSession,
) -> int:
    while True:
        try:
            message = channel.read_message()
        except JsonlProtocolError as error:
            channel.emit(
                {
                    "type": "runtime_error",
                    "code": error.code,
                    "message": str(error),
                }
            )
            continue
        if message is None:
            if not _close_application_session(channel, application_session):
                return 1
            channel.emit({"type": "runtime_closed"})
            return 0

        message_type = message.get("type")
        if message_type == "close":
            if not _close_application_session(channel, application_session):
                return 1
            channel.emit({"type": "runtime_closed"})
            return 0
        if message_type == "turn":
            if not _run_turn_message(channel, application_session, message):
                return 1
            continue
        channel.emit(
            {
                "type": "runtime_error",
                "code": "unexpected_message",
                "message": "Expected turn or close.",
            }
        )


def _run_turn_message(
    channel: JsonlChannel,
    application_session: AgentApplicationSession,
    message: dict[str, object],
) -> bool:
    turn_id = message.get("turn_id")
    content = message.get("content")
    if not isinstance(turn_id, str) or not turn_id.strip():
        channel.emit(
            {
                "type": "runtime_error",
                "code": "missing_turn_id",
                "message": "turn requires a non-empty turn_id.",
            }
        )
        return True
    if not isinstance(content, str) or not content.strip():
        channel.emit(
            {
                "type": "runtime_error",
                "code": "invalid_turn_content",
                "message": "turn content must be a string.",
            }
        )
        return True

    try:
        application_session.run_turn(
            content,
            turn_id=turn_id,
            event_handler=lambda event: channel.emit(serialize_runtime_event(event)),
        )
    except Exception as error:  # noqa: BLE001 - turn boundary
        channel.diagnostic(f"turn failed: {type(error).__name__}: {error}")
        channel.emit(
            {
                "type": "runtime_error",
                "code": "turn_failed",
                "turn_id": turn_id,
                "message": "Turn failed.",
            }
        )
        _interrupt_application_session(channel, application_session)
        return False
    return True


def _close_application_session(
    channel: JsonlChannel,
    application_session: AgentApplicationSession,
) -> bool:
    try:
        application_session.close()
    except BaseException as error:  # noqa: BLE001 - lifecycle boundary
        channel.diagnostic(
            f"runtime close failed: {type(error).__name__}: {error}"
        )
        _emit_runtime_error_safely(
            channel,
            code="lifecycle_failed",
            message="Runtime cleanup failed.",
        )
        _interrupt_application_session(channel, application_session, emit_error=False)
        return False
    return True


def _interrupt_application_session(
    channel: JsonlChannel,
    application_session: AgentApplicationSession,
    *,
    emit_error: bool = True,
) -> None:
    try:
        application_session.interrupt()
    except BaseException as error:  # noqa: BLE001 - lifecycle boundary
        channel.diagnostic(
            f"runtime cleanup failed: {type(error).__name__}: {error}"
        )
        if emit_error:
            _emit_runtime_error_safely(
                channel,
                code="lifecycle_failed",
                message="Runtime cleanup failed.",
            )


def _emit_runtime_error_safely(
    channel: JsonlChannel,
    *,
    code: str,
    message: str,
) -> None:
    try:
        channel.emit(
            {
                "type": "runtime_error",
                "code": code,
                "message": message,
            }
        )
    except BaseException as error:  # noqa: BLE001 - error reporting boundary
        channel.diagnostic(
            f"runtime error reporting failed: {type(error).__name__}: {error}"
        )


def serialize_runtime_event(event: RuntimeEvent) -> dict[str, object]:
    if event.type == "runtime_ready":
        return {
            "type": "runtime_ready",
            "session_id": event.session_id,
            "session_title": event.session_title,
            "session_created": event.session_created,
            "compact_state_recovered": event.compact_state_recovered,
            "instruction_sources": list(event.instruction_sources),
            "instruction_warnings": list(event.instruction_warnings),
            "skill_warnings": list(event.skill_warnings),
        }
    if event.type == "mcp_status":
        if event.mcp_status is None:
            raise ValueError("mcp_status event has no status")
        return {"type": "mcp_status", **serialize_mcp_status(event.mcp_status)}
    if event.type == "agent":
        if event.agent_event is None:
            raise ValueError("agent event has no AgentEvent")
        return {
            "type": "agent_event",
            "turn_id": event.turn_id,
            "event": serialize_agent_event(event.agent_event),
        }
    if event.type == "turn_finished":
        if event.outcome is None:
            raise ValueError("turn_finished event has no outcome")
        return {
            "type": "turn_finished",
            "turn_id": event.turn_id,
            "status": event.outcome.status,
            "stop_reason": event.outcome.stop_reason,
        }
    raise ValueError(f"Unsupported runtime event type: {event.type}")


def serialize_agent_event(event: AgentEvent) -> dict[str, object]:
    payload: dict[str, object] = {"type": event.type}
    if event.content:
        payload["content"] = event.content
    if event.turn_number is not None:
        payload["turn_number"] = event.turn_number
    if event.max_turns is not None:
        payload["max_turns"] = event.max_turns
    if event.progress is not None:
        payload["progress"] = _serialize_progress(event.progress)
    if event.model_retry is not None:
        payload["model_retry"] = _serialize_model_retry(event.model_retry)
    if event.tool_call is not None:
        payload["tool_call"] = _serialize_tool_call(event.tool_call)
    if event.tool_result is not None:
        payload["tool_result"] = _serialize_tool_result(event.tool_result)
    if event.stop_reason is not None:
        payload["stop_reason"] = event.stop_reason
    if event.error is not None:
        payload["error"] = event.error
    if event.type == "reasoning_state":
        payload["reasoning_state"] = event.reasoning_state
    return payload


def serialize_mcp_status(status: MCPServerStatus) -> dict[str, object]:
    return {
        "alias": status.alias,
        "status": status.status,
        "tool_count": status.tool_count,
        "error_type": status.error_type,
        "error_summary": status.error_summary,
    }


def _serialize_progress(progress: AgentProgressSnapshot) -> dict[str, object]:
    return {
        "stagnation_turns": progress.stagnation_turns,
        "same_tool_repeat": progress.same_tool_repeat,
        "same_result_repeat": progress.same_result_repeat,
        "resource_repeat": progress.resource_repeat,
        "convergence_guided": progress.convergence_guided,
        "reason": progress.reason,
    }


def _serialize_model_retry(retry: AgentModelRetry) -> dict[str, object]:
    return {
        "attempt": retry.attempt,
        "max_retries": retry.max_retries,
        "delay_seconds": retry.delay_seconds,
        "error_type": retry.error_type,
        "error_code": retry.error_code,
        "retryable": retry.retryable,
        "call_kind": retry.call_kind,
        "stream_started": retry.stream_started,
        "partial_output_chars": retry.partial_output_chars,
    }


def _serialize_tool_call(tool_call: AgentToolCall) -> dict[str, object]:
    return {
        "id": tool_call.id,
        "name": tool_call.name,
        "arguments": _json_safe(tool_call.arguments),
    }


def _serialize_tool_result(result: ToolResult) -> dict[str, object]:
    return {
        "ok": result.ok,
        "content": result.content,
        "error": result.error,
        "metadata": _json_safe(result.metadata),
    }


def _serialize_mcp_trust_server(server: MCPTrustServer) -> dict[str, object]:
    payload: dict[str, object] = {
        "alias": server.alias,
        "transport": server.transport,
    }
    if server.transport == "stdio":
        payload.update(
            {
                "command": server.command,
                "args": list(server.args),
                "env_keys": list(server.env_keys),
            }
        )
    else:
        payload.update(
            {
                "url_template": server.url_template,
                "destination": server.destination,
                "header_keys": list(server.header_keys),
            }
        )
    return payload


def _json_safe(value: object) -> object:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("JSON object keys must be strings")
            result[key] = _json_safe(item)
        return result
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    raise TypeError(f"Unsupported JSON value type: {type(value).__name__}")
