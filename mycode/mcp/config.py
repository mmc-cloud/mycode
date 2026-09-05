import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    ValidationInfo,
    field_validator,
)

from mycode.config import (
    MYCODE_CONFIG_DIR_NAME,
    load_layered_environment,
)

DEFAULT_MCP_CONFIG_FILE = "mcp.json"
DEFAULT_MCP_CONNECT_TIMEOUT_SECONDS = 10.0
DEFAULT_MCP_TOOL_TIMEOUT_SECONDS = 60.0
DEFAULT_MCP_SHUTDOWN_TIMEOUT_SECONDS = 10.0
_ALIAS_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


class MCPConfigError(ValueError):
    pass


class _ServerBase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    connect_timeout: float = Field(
        default=DEFAULT_MCP_CONNECT_TIMEOUT_SECONDS, gt=0
    )
    tool_timeout: float = Field(default=DEFAULT_MCP_TOOL_TIMEOUT_SECONDS, gt=0)


class MCPStdioServerConfig(_ServerBase):
    transport: Literal["stdio"]
    command: str = Field(min_length=1)
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict, repr=False)

    @field_validator("command")
    @classmethod
    def _non_blank_command(cls, value: str) -> str:
        if value.strip() == "":
            raise ValueError("command must not be blank")
        return value


class MCPStreamableHTTPServerConfig(_ServerBase):
    transport: Literal["streamable_http"]
    url: str = Field(min_length=1)
    headers: dict[str, str] = Field(default_factory=dict, repr=False)

    @field_validator("url")
    @classmethod
    def _http_url(cls, value: str, info: ValidationInfo) -> str:
        allow_templates = bool(
            info.context and info.context.get("allow_secret_templates")
        )
        if not value.startswith(("http://", "https://")) and not (
            allow_templates and _ENV_PATTERN.search(value)
        ):
            raise ValueError("url must use http or https")
        return value


MCPServerConfig = Annotated[
    MCPStdioServerConfig | MCPStreamableHTTPServerConfig,
    Field(discriminator="transport"),
]


class MCPConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    mcp_servers: dict[str, MCPServerConfig] = Field(
        default_factory=dict, alias="mcpServers"
    )

    @field_validator("mcp_servers")
    @classmethod
    def _valid_aliases(
        cls, value: dict[str, MCPServerConfig]
    ) -> dict[str, MCPServerConfig]:
        invalid = [alias for alias in value if _ALIAS_PATTERN.fullmatch(alias) is None]
        if invalid:
            raise ValueError("server aliases must match [A-Za-z0-9_-]{1,64}")
        return value


@dataclass(frozen=True)
class MCPLoadedConfig:
    """Resolved MCP layers plus the unresolved project layer used for trust."""

    user: MCPConfig = field(repr=False)
    project: MCPConfig = field(repr=False)
    merged: MCPConfig = field(repr=False)
    project_unresolved: MCPConfig = field(repr=False)


def default_mcp_config_file() -> Path:
    return Path.home() / MYCODE_CONFIG_DIR_NAME / DEFAULT_MCP_CONFIG_FILE


def project_mcp_config_file(workspace_root: str | Path) -> Path:
    return Path(workspace_root) / MYCODE_CONFIG_DIR_NAME / DEFAULT_MCP_CONFIG_FILE


def load_mcp_config(
    config_file: str | Path | None = None,
    *,
    env_file: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
    workspace_root: str | Path | None = None,
) -> MCPConfig:
    return load_mcp_config_layers(
        config_file,
        env_file=env_file,
        environ=environ,
        workspace_root=workspace_root,
    ).merged


def load_mcp_config_layers(
    config_file: str | Path | None = None,
    *,
    env_file: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
    workspace_root: str | Path | None = None,
) -> MCPLoadedConfig:
    user_path = default_mcp_config_file() if config_file is None else Path(config_file)
    user_config = _load_mcp_config_file(user_path)
    project_config = (
        MCPConfig()
        if workspace_root is None
        else _load_mcp_config_file(project_mcp_config_file(workspace_root))
    )
    environment = load_layered_environment(
        env_file,
        workspace_root=workspace_root,
        environ=environ,
    )
    resolved_user_config = _resolve_config_secrets(user_config, environment)
    resolved_project_config = _resolve_config_secrets(project_config, environment)
    return MCPLoadedConfig(
        user=resolved_user_config,
        project=resolved_project_config,
        merged=merge_mcp_configs(resolved_user_config, resolved_project_config),
        project_unresolved=project_config,
    )


def merge_mcp_configs(user: MCPConfig, project: MCPConfig) -> MCPConfig:
    return MCPConfig(
        mcpServers={**user.mcp_servers, **project.mcp_servers}
    )


def _resolve_config_secrets(
    config: MCPConfig,
    environment: Mapping[str, str],
) -> MCPConfig:
    resolved = _resolve_secrets(config.model_dump(by_alias=True), environment)
    try:
        return MCPConfig.model_validate(resolved)
    except ValidationError as error:
        raise MCPConfigError("Invalid MCP config: validation_error") from error


def _load_mcp_config_file(path: Path) -> MCPConfig:
    if not path.exists():
        return MCPConfig()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise MCPConfigError(f"Invalid MCP config: {type(error).__name__}") from error
    try:
        return MCPConfig.model_validate(
            payload,
            context={"allow_secret_templates": True},
        )
    except ValidationError as error:
        raise MCPConfigError("Invalid MCP config: validation_error") from error


def _resolve_secrets(value: object, environment: Mapping[str, str | None]) -> object:
    if isinstance(value, dict):
        return {key: _resolve_secrets(item, environment) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_secrets(item, environment) for item in value]
    if not isinstance(value, str):
        return value

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        replacement = environment.get(name)
        if replacement is None or replacement == "":
            raise MCPConfigError(f"Missing environment variable: {name}")
        return replacement

    return _ENV_PATTERN.sub(replace, value)
