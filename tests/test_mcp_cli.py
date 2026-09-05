from types import SimpleNamespace

import pytest

from mycode.cli import run_agent_command
from mycode.config import LLMConfig
from mycode.mcp.config import MCPConfig, MCPLoadedConfig
from mycode.mcp.models import MCPServerStatus
from mycode.session_runtime import SessionStartRequest
from mycode.session_store import SessionStore
from mycode.tools import PydanticTool, ToolArgs, ToolRegistry, ToolResult


class ExternalArgs(ToolArgs):
    value: str


class ExternalTool(PydanticTool[ExternalArgs]):
    name = "mcp__local__echo"
    description = "Echo."
    args_model = ExternalArgs
    capability = "read"
    risk = "low"

    def _run(self, args: ExternalArgs) -> ToolResult:
        return ToolResult.success(args.value)


def test_cli_displays_status_registers_snapshot_and_closes_manager(tmp_path, monkeypatch) -> None:
    manager = SimpleNamespace(
        statuses=(
            MCPServerStatus(alias="local", status="connected", tool_count=1),
            MCPServerStatus(
                alias="github",
                status="failed",
                error_type="HTTPStatusError",
                error_summary="Authentication failed (401)",
            ),
        ),
        tools=(ExternalTool(),),
        started=False,
        closed=False,
    )
    manager.start = lambda: setattr(manager, "started", True)
    manager.close = lambda: setattr(manager, "closed", True)
    monkeypatch.setattr("mycode.cli.MCPManager", lambda config, observability_sink=None: manager)
    runner = SimpleNamespace(tool_registry=ToolRegistry())
    monkeypatch.setattr("mycode.cli.build_agent_runner", lambda **kwargs: runner)
    monkeypatch.setattr("mycode.cli.run_agent_loop", lambda **kwargs: None)
    output = []

    run_agent_command(
        workspace_path=tmp_path,
        input_func=lambda prompt: "/quit",
        output_func=output.append,
        session_request=SessionStartRequest(mode="new"),
        session_store=SessionStore(tmp_path / "state.sqlite3"),
        llm_config=LLMConfig(api_key="test", base_url="https://example.test/v1", model="test"),
        mcp_config=MCPConfig(),
    )

    assert manager.started is True
    assert manager.closed is True
    assert runner.tool_registry.require("mcp__local__echo").name == "mcp__local__echo"
    assert "MCP servers:" in output
    assert any("local" in line and "1 tools" in line for line in output)
    assert any(
        "github" in line and "Authentication failed (401)" in line
        for line in output
    )


def test_agent_command_loads_mcp_config_for_workspace(tmp_path, monkeypatch) -> None:
    captured = {}

    def fake_load_mcp_config(**kwargs):
        captured.update(kwargs)
        empty = MCPConfig()
        return MCPLoadedConfig(
            user=empty,
            project=empty,
            merged=empty,
            project_unresolved=empty,
        )

    monkeypatch.setattr("mycode.cli.load_mcp_config_layers", fake_load_mcp_config)
    monkeypatch.setattr("mycode.cli.run_agent_loop", lambda **kwargs: None)

    run_agent_command(
        workspace_path=tmp_path,
        input_func=lambda prompt: "/quit",
        output_func=lambda message: None,
        session_request=SessionStartRequest(mode="new"),
        session_store=SessionStore(tmp_path / "state.sqlite3"),
        llm_config=LLMConfig(
            api_key="test",
            base_url="https://example.test/v1",
            model="test",
        ),
    )

    assert captured == {"workspace_root": tmp_path.resolve()}


def test_rejected_project_mcp_never_reaches_manager_start(tmp_path, monkeypatch) -> None:
    user = MCPConfig.model_validate(
        {
            "mcpServers": {
                "shared": {"transport": "stdio", "command": "user-command"}
            }
        }
    )
    project = MCPConfig.model_validate(
        {
            "mcpServers": {
                "shared": {"transport": "stdio", "command": "project-command"},
                "remote": {
                    "transport": "streamable_http",
                    "url": "https://example.test/mcp",
                },
            }
        }
    )
    loaded = MCPLoadedConfig(
        user=user,
        project=project,
        merged=MCPConfig(
            mcpServers={**user.mcp_servers, **project.mcp_servers}
        ),
        project_unresolved=project,
    )
    monkeypatch.setattr("mycode.cli.load_mcp_config_layers", lambda **kwargs: loaded)
    monkeypatch.setattr(
        "mycode.mcp.trust.default_mcp_trust_file",
        lambda: tmp_path / "mcp-trust.json",
    )
    captured = []
    events = []

    class CapturingManager:
        statuses = ()
        tools = ()

        def __init__(self, config, observability_sink=None):
            self.config = config

        def start(self):
            events.append("start")
            captured.append(self.config)

        def close(self):
            pass

    monkeypatch.setattr("mycode.cli.MCPManager", CapturingManager)
    loop_calls = []
    monkeypatch.setattr(
        "mycode.cli.run_agent_loop", lambda **kwargs: loop_calls.append(kwargs)
    )

    def reject(prompt):
        events.append("decision")
        return "n"

    run_agent_command(
        workspace_path=tmp_path,
        input_func=reject,
        output_func=lambda message: None,
        session_request=SessionStartRequest(mode="new"),
        session_store=SessionStore(tmp_path / "state.sqlite3"),
        llm_config=LLMConfig(
            api_key="test",
            base_url="https://example.test/v1",
            model="test",
        ),
    )

    assert len(captured) == 1
    assert events == ["decision", "start"]
    assert set(captured[0].mcp_servers) == {"shared"}
    assert captured[0].mcp_servers["shared"].command == "user-command"
    assert len(loop_calls) == 1


def test_explicit_mcp_config_bypasses_project_trust(tmp_path, monkeypatch) -> None:
    explicit = MCPConfig.model_validate(
        {
            "mcpServers": {
                "local": {"transport": "stdio", "command": "explicit-command"},
                "remote": {
                    "transport": "streamable_http",
                    "url": "https://example.test/mcp",
                },
            }
        }
    )
    captured = []

    class CapturingManager:
        statuses = ()
        tools = ()

        def __init__(self, config, observability_sink=None):
            self.config = config

        def start(self):
            captured.append(self.config)

        def close(self):
            pass

    monkeypatch.setattr("mycode.cli.MCPManager", CapturingManager)
    monkeypatch.setattr(
        "mycode.cli.load_mcp_config_layers",
        lambda **kwargs: pytest.fail("explicit config must bypass auto loading"),
    )
    monkeypatch.setattr(
        "mycode.cli.apply_project_mcp_trust",
        lambda *args, **kwargs: pytest.fail("explicit config must bypass trust"),
    )
    monkeypatch.setattr("mycode.cli.run_agent_loop", lambda **kwargs: None)

    run_agent_command(
        workspace_path=tmp_path,
        input_func=lambda prompt: pytest.fail("must not prompt"),
        output_func=lambda message: None,
        session_request=SessionStartRequest(mode="new"),
        session_store=SessionStore(tmp_path / "state.sqlite3"),
        llm_config=LLMConfig(
            api_key="test",
            base_url="https://example.test/v1",
            model="test",
        ),
        mcp_config=explicit,
    )

    assert captured == [explicit]
