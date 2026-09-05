import hashlib
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Literal
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


def apply_project_mcp_trust(
    loaded: MCPLoadedConfig,
    project: ProjectIdentity,
    *,
    input_func: Callable[[str], str] = input,
    output_func: Callable[[str], None] = print,
    trust_file: str | Path | None = None,
) -> MCPConfig:
    fingerprint = project_mcp_fingerprint(loaded)
    if fingerprint is None:
        return loaded.merged

    path = default_mcp_trust_file() if trust_file is None else Path(trust_file)
    store, invalid = _read_trust_store(path)
    if invalid:
        output_func(
            "MCP trust> 警告：信任状态文件无效；"
            "项目级 MCP 将按未信任处理"
        )
    if store.projects.get(project.key) == fingerprint:
        return loaded.merged

    _show_trust_request(loaded, output_func)
    try:
        approved = input_func(
            "MCP trust> 是否信任并启用这些项目级 MCP？[y/N] "
        ).strip().lower() in {"y", "yes"}
    except EOFError:
        approved = False

    if not approved:
        output_func("MCP trust> 未启用项目级 MCP")
        return merge_mcp_configs(loaded.user, MCPConfig())

    store.projects[project.key] = fingerprint
    try:
        _write_trust_store(path, store)
    except OSError:
        output_func(
            "MCP trust> 警告：信任状态未能保存；"
            "本次仍会启用，下次将再次询问"
        )
    return loaded.merged


def _show_trust_request(
    loaded: MCPLoadedConfig,
    output_func: Callable[[str], None],
) -> None:
    output_func("MCP trust> 项目配置请求启用以下 MCP Server：")
    for alias, server in loaded.project_unresolved.mcp_servers.items():
        output_func("")
        output_func(f"server> {alias!r}")
        output_func(f"transport> {server.transport}")
        if isinstance(server, MCPStdioServerConfig):
            output_func(f"command> {server.command!r}")
            output_func(f"args> {server.args!r}")
            if server.env:
                output_func(f"env keys> {sorted(server.env)!r}")
            continue

        resolved = loaded.project.mcp_servers[alias]
        if not isinstance(resolved, MCPStreamableHTTPServerConfig):
            raise TypeError("MCP project config layers do not match")
        output_func(f"url template> {_safe_url_template(server.url)!r}")
        destination = _safe_http_destination(resolved.url)
        if destination is not None:
            output_func(f"destination> {destination!r}")
        if server.headers:
            output_func(f"header keys> {sorted(server.headers)!r}")


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
