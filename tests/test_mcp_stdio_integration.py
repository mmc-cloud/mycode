import sys
from pathlib import Path

from mycode.agent import AgentToolCall
from mycode.mcp.config import MCPConfig
from mycode.mcp.manager import MCPManager
from mycode.runner import execute_tool_batch
from mycode.tools import ToolRegistry


def test_local_stdio_discovery_call_error_timeout_and_cleanup() -> None:
    server = Path(__file__).parent / "fixtures" / "mcp_test_server.py"
    config = MCPConfig.model_validate({"mcpServers": {"local": {
        "transport": "stdio",
        "command": sys.executable,
        "args": [str(server)],
        "connect_timeout": 10,
        "tool_timeout": 0.1,
    }}})
    manager = MCPManager(config)
    manager.start()
    try:
        assert manager.statuses[0].status == "connected"
        registry = ToolRegistry.from_tools(manager.tools)
        names = [tool.name for tool in manager.tools]
        assert names == [
            "mcp__local__echo", "mcp__local__add",
            "mcp__local__fail", "mcp__local__slow",
        ]
        batch = execute_tool_batch(registry, [
            AgentToolCall(id="1", name="mcp__local__echo", arguments={"text": "hi"}),
            AgentToolCall(id="2", name="mcp__local__add", arguments={"a": 2, "b": 3}),
            AgentToolCall(id="3", name="mcp__local__fail", arguments={}),
            AgentToolCall(id="4", name="mcp__local__slow", arguments={}),
        ])
        assert batch.executions[0].result.content == "hi"
        assert batch.executions[1].result.metadata["structured_content"] == {"sum": 5}
        assert batch.executions[2].result.ok is False
        assert batch.executions[3].result.ok is False
        assert batch.executions[3].result.metadata["error_type"] == "TimeoutError"
    finally:
        manager.close()
    assert manager._clients == {}
