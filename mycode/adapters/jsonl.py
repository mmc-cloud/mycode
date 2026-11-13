"""Versioned JSONL adapter for the application runtime."""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, TextIO
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from mycode.agent.events import (
    AgentEvent,
    AgentModelRetry,
    AgentProgressSnapshot,
    AgentToolCall,
)
from mycode.application import (
    AgentApplicationSession,
    ApplicationStartupWarning,
    create_application_environment,
    prepare_application_session_factory,
)
from mycode.application.events import RuntimeEvent
from mycode.application.sessions import SessionStartRequest
from mycode.config import LLMConfig
from mycode.mcp.config import MCPConfig
from mycode.mcp.models import MCPServerStatus
from mycode.mcp.trust import MCPTrustRequest, MCPTrustServer, MCPTrustWarning
from mycode.permissions import (
    ConfirmationRequest,
    ConfirmationResult,
    Confirmer,
)
from mycode.persistence.connection_store import ConnectionStoreError
from mycode.persistence.session_store import (
    SessionInUseError,
    SessionNotFoundError,
    SessionStore,
    SessionStoreError,
)
from mycode.tools.base import ToolResult


JSONL_PROTOCOL_VERSION = 1
_PERMISSION_DECISIONS = {"once", "task", "session", "reject"}


class ModelMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, hide_input_in_errors=True)
    version: Literal[1]
    type: Literal["model"]
    request_id: str = Field(min_length=1, max_length=200)
    action: Literal["status", "set", "default", "clear_default"] = "status"
    role: Literal["main", "subagent", "compact"] | None = None
    model: str | None = Field(default=None, min_length=1, max_length=200)
    connection: str | None = Field(default=None, min_length=1, max_length=200)
    thinking: Literal["auto", "on", "off"] | None = None
    effort: Literal["auto", "low", "high", "max"] | None = None
    window: int | None = Field(default=None, ge=1)
    reserve: int | None = Field(default=None, ge=0)
    margin: int | None = Field(default=None, ge=0)
    output: Annotated[int, Field(ge=1)] | Literal["auto"] | None = None

    @model_validator(mode="after")
    def action_fields(self):
        parameters = self.model_fields_set - {"version", "type", "request_id", "action"}
        if self.action == "set":
            if self.role is None or "model" not in parameters:
                raise ValueError("set requires role and model (null follows Main).")
        elif parameters:
            raise ValueError("Only set accepts model parameters.")
        return self


class ConnectMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, hide_input_in_errors=True)
    version: Literal[1]
    type: Literal["connect"]
    request_id: str = Field(min_length=1, max_length=200)
    action: Literal["list", "add", "edit", "delete", "default", "models", "use"] = "list"
    id: str | None = Field(default=None, min_length=1, max_length=200)
    model: str | None = Field(default=None, min_length=1, max_length=200)
    base_url: str | None = Field(default=None, min_length=1)
    api_key: str | None = Field(default=None, min_length=1, repr=False)
    headers: dict[str, str] | None = Field(default=None, repr=False)

    @model_validator(mode="after")
    def action_fields(self):
        fields = self.model_fields_set - {"version", "type", "request_id", "action"}
        allowed = ({"id", "base_url", "api_key", "headers"} if self.action in {"add", "edit"}
                   else {"id", "model"} if self.action == "use"
                   else set() if self.action == "list" else {"id"})
        if fields - allowed or (self.action != "list" and self.id is None):
            raise ValueError("Invalid fields for connection action.")
        if self.action == "add" and (self.base_url is None or self.api_key is None):
            raise ValueError("add requires base_url and api_key.")
        if self.action == "edit" and not fields & {"base_url", "api_key", "headers"}:
            raise ValueError("edit requires connection parameters.")
        if self.action == "use" and self.model is None:
            raise ValueError("use requires model.")
        if any(getattr(self, name) is None for name in fields - {"id"}):
            raise ValueError("Connection parameters cannot be null.")
        return self


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
            if message.get("type") in {"model", "connect"}:
                _management_error(self.channel, message, "runtime_busy", "Wait for the active request to finish.")
                continue
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
            if message.get("type") in {"model", "connect"}:
                _management_error(self.channel, message, "runtime_busy", "Wait for startup to finish.")
                continue
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
    environment = create_application_environment(
        workspace_path,
        session_store=session_store,
    )
    request = (
        SessionStartRequest(mode="continue")
        if session_request is None
        else session_request
    )

    confirmer = JsonlConfirmer(channel)
    trust_confirmer = JsonlMCPTrustConfirmer(channel)

    def handle_startup_warning(warning: ApplicationStartupWarning) -> None:
        channel.emit(
            {
                "type": "runtime_error",
                "code": warning.code,
                "message": warning.message,
            }
        )

    try:
        factory = prepare_application_session_factory(
            environment,
            llm_config=llm_config,
            mcp_config=mcp_config,
            confirmer=confirmer,
            mcp_trust_confirmer=trust_confirmer,
            warning_handler=handle_startup_warning,
        )
        application_session = factory.open_session(request)
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
        if message_type == "context_status":
            _run_context_status_message(channel, application_session)
            continue
        if message_type == "compact":
            _run_compact_message(channel, application_session)
            continue
        if message_type in {"model", "connect"}:
            _run_management_message(channel, application_session, message)
            continue
        channel.emit(
            {
                "type": "runtime_error",
                "code": "unexpected_message",
                "message": "Expected turn, context_status, compact, model, connect, or close.",
            }
        )


def _management_error(channel, message, code, detail):
    request_id = message.get("request_id")
    channel.emit({"type": message["type"] + "_result",
                  "request_id": request_id if isinstance(request_id, str) else None,
                  "ok": False, "error": {"code": code, "message": detail}})


def _model_state(manager):
    roles = {}
    for role in ("main", "subagent", "compact"):
        config = manager.runner.model_configuration.resolve(role)
        window = manager.windows[role]
        roles[role] = {
            "selection": getattr(manager.selection, role).model_dump(exclude={"window_binding"}),
            "connection": manager.selection.identity(role)[0], "model": config.model,
            "thinking": config.thinking_enabled, "effort": config.reasoning_effort,
            "window": window.tokens, "window_source": window.source,
            "server_limit": window.server_limit, "reserve": config.reserved_output_tokens,
            "margin": config.context_safety_margin_tokens, "output": config.max_output_tokens,
        }
    return roles


def _run_management_message(channel, application_session, message):
    try:
        request = (ModelMessage if message["type"] == "model" else ConnectMessage).model_validate(message)
        manager = application_session.models
        if isinstance(request, ModelMessage):
            if request.action == "set":
                options = request.model_dump(include=request.model_fields_set - {
                    "version", "type", "request_id", "action", "role", "model"})
                manager.select_model(request.role, request.model, **options)
            elif request.action in {"default", "clear_default"}:
                manager.command("model", ("default", "clear") if request.action == "clear_default" else ("default",))
            data = {"roles": _model_state(manager)}
        elif request.action == "models":
            rows = manager.discover_models(request.id)
            data = {"models": rows or [], "available": rows is not None,
                    "warning": "model_list_unavailable" if rows is None else None}
        else:
            if request.action == "use":
                manager.select_model("main", request.model, connection=request.id)
            elif request.action != "list":
                values = request.model_dump(include={"base_url", "api_key", "headers"}, exclude_unset=True)
                manager.command("connect", (request.action, request.id), values=values)
            doc = manager.store.read()
            data = {"default_connection": (doc.default_models.main.connection_ref
                        if doc.default_models is not None else doc.default_ref or "env"),
                    "connections": [{"id": "env", "read_only": True}, *(
                        {"id": key, "base_url": value.base_url, "read_only": False}
                        for key, value in sorted(doc.connections.items()))]}
    except ValidationError:
        _management_error(channel, message, "invalid_parameters", "Invalid fields or parameter types for this action.")
        return
    except ConnectionStoreError as error:
        # These domain errors already contain credential-free user messages.
        _management_error(channel, message, "configuration_rejected", str(error))
        return
    except ValueError:
        _management_error(channel, message, "configuration_rejected", "Check connection references, model parameters, storage and history budgets.")
        return
    except Exception:  # Adapter boundary: SDK/validation exceptions may contain credentials.
        _management_error(channel, message, "operation_failed", "Operation failed; retry when idle and check connection/storage availability.")
        return
    channel.emit({"type": request.type + "_result", "request_id": request.request_id,
                  "ok": True, "action": request.action, **data})


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


def _run_context_status_message(
    channel: JsonlChannel,
    application_session: AgentApplicationSession,
) -> None:
    try:
        status = application_session.get_context_status()
    except Exception as error:  # noqa: BLE001 - control boundary
        channel.diagnostic(
            f"context status failed: {type(error).__name__}: {error}"
        )
        channel.emit(
            {
                "type": "runtime_error",
                "code": "context_status_failed",
                "message": "Context status inspection failed.",
            }
        )
        return
    channel.emit({"type": "context_status", **status.to_dict()})


def _run_compact_message(
    channel: JsonlChannel,
    application_session: AgentApplicationSession,
) -> None:
    try:
        result = application_session.compact_context()
    except Exception as error:  # noqa: BLE001 - control boundary
        channel.diagnostic(f"compact failed: {type(error).__name__}: {error}")
        channel.emit(
            {
                "type": "runtime_error",
                "code": "compact_failed",
                "message": "Context Compact failed.",
            }
        )
        return
    channel.emit({"type": "compact_result", **result.to_dict()})


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
        result_payload = _serialize_tool_result(event.tool_result)
        if event.tool_call_id is not None:
            result_payload["tool_call_id"] = event.tool_call_id
        payload["tool_result"] = result_payload
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
