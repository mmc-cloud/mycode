"""Presentation-neutral project MCP trust resolution."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol
from urllib.parse import urlsplit
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from mycode.config import MYCODE_CONFIG_DIR_NAME
from mycode.mcp.config import (
    MCPConfig,
    MCPLoadedConfig,
    MCPStdioServerConfig,
    MCPStreamableHTTPServerConfig,
    merge_mcp_configs,
)
from mycode.project import ProjectIdentity


DEFAULT_MCP_TRUST_FILE = "mcp-trust.json"


class MCPTrustStore(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: Literal[1] = 1
    projects: dict[str, str] = Field(default_factory=dict)

    @field_validator("projects")
    @classmethod
    def _valid_digests(cls, value: dict[str, str]) -> dict[str, str]:
        if any(
            not _is_sha256(key) or not _is_sha256(digest)
            for key, digest in value.items()
        ):
            raise ValueError("project keys and fingerprints must be SHA-256 digests")
        return value


MCPTrustTransport = Literal["stdio", "streamable_http"]
MCPTrustWarningCode = Literal["invalid_trust_store", "persistence_failed"]


@dataclass(frozen=True)
class MCPTrustServer:
    """Safe, displayable project MCP information without secret values."""

    alias: str
    transport: MCPTrustTransport
    command: str | None = None
    args: tuple[str, ...] = ()
    env_keys: tuple[str, ...] = ()
    url_template: str | None = None
    destination: str | None = None
    header_keys: tuple[str, ...] = ()


@dataclass(frozen=True)
class MCPTrustRequest:
    servers: tuple[MCPTrustServer, ...]


@dataclass(frozen=True)
class MCPTrustWarning:
    code: MCPTrustWarningCode
    message: str


@dataclass(frozen=True)
class MCPTrustResolution:
    config: MCPConfig
    approved: bool | None = None
    warnings: tuple[MCPTrustWarning, ...] = ()


class MCPTrustConfirmer(Protocol):
    def confirm(self, request: MCPTrustRequest) -> bool:
        pass


def default_mcp_trust_file() -> Path:
    return Path.home() / MYCODE_CONFIG_DIR_NAME / DEFAULT_MCP_TRUST_FILE


def project_mcp_fingerprint(loaded: MCPLoadedConfig) -> str | None:
    entries: list[dict[str, object]] = []
    for alias in sorted(loaded.project.mcp_servers):
        resolved = loaded.project.mcp_servers[alias]
        unresolved = loaded.project_unresolved.mcp_servers[alias]
        if isinstance(resolved, MCPStdioServerConfig) and isinstance(
            unresolved, MCPStdioServerConfig
        ):
            entries.append(
                {
                    "alias": alias,
                    "transport": "stdio",
                    "command": {
                        "unresolved": unresolved.command,
                        "resolved": resolved.command,
                    },
                    "args": {
                        "unresolved": unresolved.args,
                        "resolved": resolved.args,
                    },
                    "env": unresolved.env,
                    "connect_timeout": resolved.connect_timeout,
                    "tool_timeout": resolved.tool_timeout,
                }
            )
            continue
        if isinstance(resolved, MCPStreamableHTTPServerConfig) and isinstance(
            unresolved, MCPStreamableHTTPServerConfig
        ):
            entries.append(
                {
                    "alias": alias,
                    "transport": "streamable_http",
                    "url": {
                        "unresolved": unresolved.url,
                        "resolved": resolved.url,
                    },
                    "headers": unresolved.headers,
                    "connect_timeout": resolved.connect_timeout,
                    "tool_timeout": resolved.tool_timeout,
                }
            )
            continue
        raise TypeError("MCP project config layers do not match")
    if not entries:
        return None
    canonical = json.dumps(
        {"version": 1, "servers": entries},
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def resolve_project_mcp_trust(
    loaded: MCPLoadedConfig,
    project: ProjectIdentity,
    *,
    confirmer: MCPTrustConfirmer,
    trust_file: str | Path | None = None,
) -> MCPTrustResolution:
    fingerprint = project_mcp_fingerprint(loaded)
    if fingerprint is None:
        return MCPTrustResolution(config=loaded.merged)

    path = default_mcp_trust_file() if trust_file is None else Path(trust_file)
    store, invalid = _read_trust_store(path)
    warnings: list[MCPTrustWarning] = []
    if invalid:
        warning = MCPTrustWarning(
            code="invalid_trust_store",
            message=(
                "MCP trust store is invalid; project MCP will be treated as untrusted."
            ),
        )
        warnings.append(warning)
        _report_warning(confirmer, warning)
    if store.projects.get(project.key) == fingerprint:
        return MCPTrustResolution(config=loaded.merged, warnings=tuple(warnings))

    approved = confirmer.confirm(_build_trust_request(loaded))
    if not approved:
        return MCPTrustResolution(
            config=merge_mcp_configs(loaded.user, MCPConfig()),
            approved=False,
            warnings=tuple(warnings),
        )

    store.projects[project.key] = fingerprint
    try:
        _write_trust_store(path, store)
    except OSError:
        warning = MCPTrustWarning(
            code="persistence_failed",
            message=(
                "MCP trust store could not be saved; this run remains enabled."
            ),
        )
        warnings.append(warning)
        _report_warning(confirmer, warning)
    return MCPTrustResolution(
        config=loaded.merged,
        approved=True,
        warnings=tuple(warnings),
    )


def apply_project_mcp_trust(
    loaded: MCPLoadedConfig,
    project: ProjectIdentity,
    *,
    confirmer: MCPTrustConfirmer,
    trust_file: str | Path | None = None,
) -> MCPConfig:
    """Compatibility name for callers that only need the resolved config."""

    return resolve_project_mcp_trust(
        loaded,
        project,
        confirmer=confirmer,
        trust_file=trust_file,
    ).config


def _build_trust_request(loaded: MCPLoadedConfig) -> MCPTrustRequest:
    servers: list[MCPTrustServer] = []
    for alias in sorted(loaded.project_unresolved.mcp_servers):
        unresolved = loaded.project_unresolved.mcp_servers[alias]
        resolved = loaded.project.mcp_servers[alias]
        if isinstance(unresolved, MCPStdioServerConfig) and isinstance(
            resolved, MCPStdioServerConfig
        ):
            servers.append(
                MCPTrustServer(
                    alias=alias,
                    transport="stdio",
                    command=unresolved.command,
                    args=tuple(unresolved.args),
                    env_keys=tuple(sorted(unresolved.env)),
                )
            )
            continue
        if isinstance(unresolved, MCPStreamableHTTPServerConfig) and isinstance(
            resolved, MCPStreamableHTTPServerConfig
        ):
            servers.append(
                MCPTrustServer(
                    alias=alias,
                    transport="streamable_http",
                    url_template=_safe_url_template(unresolved.url),
                    destination=_safe_http_destination(resolved.url),
                    header_keys=tuple(sorted(unresolved.headers)),
                )
            )
            continue
        raise TypeError("MCP project config layers do not match")
    return MCPTrustRequest(servers=tuple(servers))


def _report_warning(confirmer: MCPTrustConfirmer, warning: MCPTrustWarning) -> None:
    reporter = getattr(confirmer, "report_warning", None)
    if reporter is not None:
        reporter(warning)


def _safe_url_template(url: str) -> str:
    if url.startswith("${") and url.endswith("}") and url.count("${") == 1:
        return url
    sanitized = _safe_http_url(url, include_path=True)
    return "<unavailable>" if sanitized is None else sanitized


def _safe_http_destination(url: str) -> str | None:
    return _safe_http_url(url, include_path=False)


def _safe_http_url(url: str, *, include_path: bool) -> str | None:
    try:
        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https"} or parsed.hostname is None:
            return None
        host = parsed.hostname
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        port = parsed.port
    except (TypeError, ValueError):
        return None
    authority = host if port is None else f"{host}:{port}"
    path = parsed.path if include_path else ""
    return f"{parsed.scheme}://{authority}{path}"


def _read_trust_store(path: Path) -> tuple[MCPTrustStore, bool]:
    try:
        if not path.exists():
            return MCPTrustStore(), False
        payload = json.loads(path.read_text(encoding="utf-8"))
        return MCPTrustStore.model_validate(payload), False
    except (OSError, UnicodeError, json.JSONDecodeError, ValidationError):
        return MCPTrustStore(), True


def _write_trust_store(path: Path, store: MCPTrustStore) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(store.model_dump(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )
