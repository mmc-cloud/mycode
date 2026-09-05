import json
from pathlib import Path

import pytest

from mycode.mcp.config import (
    MCPConfigError,
    load_mcp_config,
    load_mcp_config_layers,
)


def write_mcp(path: Path, servers: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"mcpServers": servers}), encoding="utf-8")


def stdio_server(command: str, *, token: str | None = None) -> dict[str, object]:
    server: dict[str, object] = {"transport": "stdio", "command": command}
    if token is not None:
        server["env"] = {"TOKEN": token}
    return server


def test_missing_user_and_project_mcp_config_disables_mcp(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    config = load_mcp_config(
        tmp_path / "missing-user.json",
        env_file=tmp_path / "missing-user.env",
        environ={},
        workspace_root=workspace,
    )

    assert config.mcp_servers == {}


def test_loads_user_config_without_project_config(tmp_path) -> None:
    user_path = tmp_path / "user.json"
    write_mcp(user_path, {"user": stdio_server("user-command")})

    config = load_mcp_config(
        user_path,
        env_file=tmp_path / "missing-user.env",
        environ={},
        workspace_root=tmp_path / "workspace",
    )

    assert config.mcp_servers["user"].command == "user-command"


def test_loads_project_config_without_user_config(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    write_mcp(
        workspace / ".mycode" / "mcp.json",
        {"project": stdio_server("project-command")},
    )

    config = load_mcp_config(
        tmp_path / "missing-user.json",
        env_file=tmp_path / "missing-user.env",
        environ={},
        workspace_root=workspace,
    )

    assert config.mcp_servers["project"].command == "project-command"


def test_merges_aliases_and_project_alias_overrides_user(tmp_path) -> None:
    user_path = tmp_path / "user.json"
    write_mcp(
        user_path,
        {
            "user": stdio_server("user-command"),
            "shared": stdio_server("user-shared"),
        },
    )
    workspace = tmp_path / "workspace"
    write_mcp(
        workspace / ".mycode" / "mcp.json",
        {
            "project": stdio_server("project-command"),
            "shared": stdio_server("project-shared"),
        },
    )

    config = load_mcp_config(
        user_path,
        env_file=tmp_path / "missing-user.env",
        environ={},
        workspace_root=workspace,
    )

    assert set(config.mcp_servers) == {"user", "project", "shared"}
    assert config.mcp_servers["shared"].command == "project-shared"


def test_layered_loader_preserves_resolved_sources_and_compat_merged_api(tmp_path) -> None:
    user_path = tmp_path / "user.json"
    write_mcp(user_path, {"shared": stdio_server("user-command")})
    workspace = tmp_path / "workspace"
    write_mcp(
        workspace / ".mycode" / "mcp.json",
        {"shared": stdio_server("${COMMAND}", token="${TOKEN}")},
    )

    loaded = load_mcp_config_layers(
        user_path,
        env_file=tmp_path / "missing.env",
        environ={"COMMAND": "project-command", "TOKEN": "secret"},
        workspace_root=workspace,
    )
    compatible = load_mcp_config(
        user_path,
        env_file=tmp_path / "missing.env",
        environ={"COMMAND": "project-command", "TOKEN": "secret"},
        workspace_root=workspace,
    )

    assert loaded.user.mcp_servers["shared"].command == "user-command"
    assert loaded.project.mcp_servers["shared"].command == "project-command"
    assert loaded.project_unresolved.mcp_servers["shared"].command == "${COMMAND}"
    assert loaded.merged == compatible


def test_project_mcp_resolves_user_env_secret(tmp_path) -> None:
    user_env = tmp_path / "user.env"
    user_env.write_text("TOKEN=user-secret\n", encoding="utf-8")
    workspace = tmp_path / "workspace"
    write_mcp(
        workspace / ".mycode" / "mcp.json",
        {"project": stdio_server("python", token="${TOKEN}")},
    )

    config = load_mcp_config(
        tmp_path / "missing-user.json",
        env_file=user_env,
        environ={},
        workspace_root=workspace,
    )

    assert config.mcp_servers["project"].env == {"TOKEN": "user-secret"}


def test_project_env_secret_overrides_user_env(tmp_path) -> None:
    user_env = tmp_path / "user.env"
    user_env.write_text("TOKEN=user-secret\n", encoding="utf-8")
    workspace = tmp_path / "workspace"
    write_mcp(
        workspace / ".mycode" / "mcp.json",
        {"project": stdio_server("python", token="${TOKEN}")},
    )
    project_env = workspace / ".mycode" / ".env"
    project_env.write_text("TOKEN=project-secret\n", encoding="utf-8")

    config = load_mcp_config(
        tmp_path / "missing-user.json",
        env_file=user_env,
        environ={},
        workspace_root=workspace,
    )

    assert config.mcp_servers["project"].env == {"TOKEN": "project-secret"}


def test_process_secret_overrides_project_and_user_env(tmp_path) -> None:
    user_path = tmp_path / "user.json"
    write_mcp(user_path, {"user": stdio_server("python", token="${TOKEN}")})
    user_env = tmp_path / "user.env"
    user_env.write_text("TOKEN=user-secret\n", encoding="utf-8")
    workspace = tmp_path / "workspace"
    project_env = workspace / ".mycode" / ".env"
    project_env.parent.mkdir(parents=True)
    project_env.write_text("TOKEN=project-secret\n", encoding="utf-8")

    config = load_mcp_config(
        user_path,
        env_file=user_env,
        environ={"TOKEN": "process-secret"},
        workspace_root=workspace,
    )

    assert config.mcp_servers["user"].env == {"TOKEN": "process-secret"}


def test_empty_process_secret_does_not_clear_project_or_user_env(tmp_path) -> None:
    user_path = tmp_path / "user.json"
    write_mcp(user_path, {"user": stdio_server("python", token="${TOKEN}")})
    user_env = tmp_path / "user.env"
    user_env.write_text("TOKEN=user-secret\n", encoding="utf-8")
    workspace = tmp_path / "workspace"
    project_env = workspace / ".mycode" / ".env"
    project_env.parent.mkdir(parents=True)
    project_env.write_text("TOKEN=project-secret\n", encoding="utf-8")

    config = load_mcp_config(
        user_path,
        env_file=user_env,
        environ={"TOKEN": "  "},
        workspace_root=workspace,
    )

    assert config.mcp_servers["user"].env == {"TOKEN": "project-secret"}


def test_loads_explicit_transports_and_resolves_secrets(tmp_path) -> None:
    path = tmp_path / "mcp.json"
    write_mcp(
        path,
        {
            "local": {
                "transport": "stdio",
                "command": "python",
                "args": ["server.py"],
                "env": {"TOKEN": "${TOKEN}"},
            },
            "remote": {
                "transport": "streamable_http",
                "url": "https://example.test/mcp",
                "headers": {"Authorization": "Bearer ${TOKEN}"},
                "connect_timeout": 2,
                "tool_timeout": 3,
            },
        },
    )

    config = load_mcp_config(
        path,
        env_file=tmp_path / "missing.env",
        environ={"TOKEN": "secret"},
    )

    assert config.mcp_servers["local"].env == {"TOKEN": "secret"}
    assert config.mcp_servers["remote"].headers == {
        "Authorization": "Bearer secret"
    }
    assert "secret" not in repr(config)
    assert config.mcp_servers["remote"].connect_timeout == 2
    assert config.mcp_servers["remote"].tool_timeout == 3


def test_project_http_url_can_be_entirely_resolved_from_environment(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    write_mcp(
        workspace / ".mycode" / "mcp.json",
        {
            "remote": {
                "transport": "streamable_http",
                "url": "${MCP_URL}",
            }
        },
    )

    loaded = load_mcp_config_layers(
        tmp_path / "missing-user.json",
        env_file=tmp_path / "missing.env",
        environ={"MCP_URL": "https://api.example.test/mcp?token=secret"},
        workspace_root=workspace,
    )

    assert loaded.project_unresolved.mcp_servers["remote"].url == "${MCP_URL}"
    assert (
        loaded.project.mcp_servers["remote"].url
        == "https://api.example.test/mcp?token=secret"
    )


@pytest.mark.parametrize("scope", ["user", "project"])
@pytest.mark.parametrize(
    "payload",
    [
        {"mcpServers": {"bad alias": stdio_server("python")}},
        {"mcpServers": {"x": {"command": "python"}}},
        {"mcpServers": {"x": {"transport": "sse", "url": "https://x"}}},
    ],
)
def test_rejects_invalid_user_or_project_config(tmp_path, scope, payload) -> None:
    user_path = tmp_path / "user.json"
    workspace = tmp_path / "workspace"
    target = user_path if scope == "user" else workspace / ".mycode" / "mcp.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(MCPConfigError, match="Invalid MCP config"):
        load_mcp_config(
            user_path,
            env_file=tmp_path / "missing.env",
            environ={},
            workspace_root=workspace,
        )


@pytest.mark.parametrize("scope", ["user", "project"])
def test_rejects_invalid_json_in_user_or_project_config(tmp_path, scope) -> None:
    user_path = tmp_path / "user.json"
    workspace = tmp_path / "workspace"
    target = user_path if scope == "user" else workspace / ".mycode" / "mcp.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("{not-json", encoding="utf-8")

    with pytest.raises(MCPConfigError, match="Invalid MCP config"):
        load_mcp_config(
            user_path,
            env_file=tmp_path / "missing.env",
            environ={},
            workspace_root=workspace,
        )


def test_rejects_missing_secret_without_echoing_values(tmp_path) -> None:
    path = tmp_path / "mcp.json"
    write_mcp(
        path,
        {"x": stdio_server("python", token="prefix-${MISSING}-private-value")},
    )

    with pytest.raises(
        MCPConfigError,
        match="Missing environment variable: MISSING",
    ) as exc:
        load_mcp_config(path, env_file=tmp_path / "missing.env", environ={})

    assert "private-value" not in str(exc.value)


def test_project_alias_override_does_not_hide_invalid_user_secret(tmp_path) -> None:
    user_path = tmp_path / "user.json"
    write_mcp(
        user_path,
        {"shared": stdio_server("python", token="${MISSING}")},
    )
    workspace = tmp_path / "workspace"
    write_mcp(
        workspace / ".mycode" / "mcp.json",
        {"shared": stdio_server("project-command")},
    )

    with pytest.raises(MCPConfigError, match="Missing environment variable: MISSING"):
        load_mcp_config(
            user_path,
            env_file=tmp_path / "missing.env",
            environ={},
            workspace_root=workspace,
        )
