import asyncio
from contextlib import asynccontextmanager
from time import monotonic

from mcp.types import (
    CallToolResult,
    ListToolsResult,
    TextContent,
    Tool,
    ToolAnnotations,
)

from mycode.mcp.config import MCPConfig
from mycode.mcp.manager import MCPManager


class FakeClient:
    def __init__(self, alias: str, closed: list[str]) -> None:
        self.alias = alias
        self.closed = closed
        self.cursors = []

    async def list_tools(self, *, cursor=None, cache_mode="use"):
        self.cursors.append(cursor)
        if self.alias == "broken":
            raise ConnectionError("offline")
        tool = Tool(
            name="one" if cursor is None else "two",
            inputSchema={"type": "object"},
            annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False),
        )
        return ListToolsResult(tools=[tool], nextCursor="next" if cursor is None else None)

    async def call_tool(self, name, arguments):
        return CallToolResult(content=[TextContent(text=f"{name}:{arguments['x']}")])


def test_best_effort_pagination_call_observability_and_cleanup(monkeypatch) -> None:
    closed = []
    clients = {}

    @asynccontextmanager
    async def fake_open(config):
        alias = config.command
        client = FakeClient(alias, closed)
        clients[alias] = client
        try:
            yield client
        finally:
            closed.append(alias)

    monkeypatch.setattr("mycode.mcp.manager.open_mcp_client", fake_open)
    events = []
    config = MCPConfig.model_validate({"mcpServers": {
        "good": {"transport": "stdio", "command": "good"},
        "broken": {"transport": "stdio", "command": "broken"},
    }})
    manager = MCPManager(config, observability_sink=events.append)
    manager.start()
    try:
        assert [(s.alias, s.status, s.tool_count) for s in manager.statuses] == [
            ("good", "connected", 2), ("broken", "failed", 0)
        ]
        broken = manager.statuses[1]
        assert broken.error_type == "ConnectionError"
        assert broken.error_summary == "Connection failed"
        assert [tool.name for tool in manager.tools] == ["mcp__good__one", "mcp__good__two"]
        assert clients["good"].cursors == [None, "next"]
        result = asyncio.run(manager.call_tool("good", "one", {"x": 7}))
        assert result.content[0].text == "one:7"
        assert any(event["event_type"] == "mcp_tool_call" for event in events)
        failed_start = next(
            event
            for event in events
            if event["event_type"] == "mcp_server_start"
            and event["server_alias"] == "broken"
        )
        assert failed_start["error_type"] == "ConnectionError"
        assert "error_summary" not in failed_start
        assert "arguments" not in repr(events)
    finally:
        manager.close()
    assert sorted(closed) == ["broken", "good"]
    assert manager.shutdown_status.status == "completed"
    assert manager._thread is None
    assert manager._loop is None
    assert manager._stop is None


def test_server_startup_runs_in_parallel(monkeypatch) -> None:
    class SlowClient:
        async def list_tools(self, *, cursor=None, cache_mode="use"):
            await asyncio.sleep(0.2)
            return ListToolsResult(tools=[], nextCursor=None)

    @asynccontextmanager
    async def slow_open(config):
        await asyncio.sleep(0.2)
        yield SlowClient()

    monkeypatch.setattr("mycode.mcp.manager.open_mcp_client", slow_open)
    config = MCPConfig.model_validate({"mcpServers": {
        "first": {
            "transport": "stdio", "command": "first",
            "connect_timeout": 0.5, "tool_timeout": 0.5,
        },
        "second": {
            "transport": "stdio", "command": "second",
            "connect_timeout": 0.5, "tool_timeout": 0.5,
        },
    }})
    manager = MCPManager(config)

    started = monotonic()
    manager.start()
    elapsed = monotonic() - started
    try:
        assert [status.status for status in manager.statuses] == [
            "connected", "connected"
        ]
        assert elapsed < 0.7
    finally:
        manager.close()


def test_connect_timeout_does_not_cover_discovery(monkeypatch) -> None:
    class DiscoveryClient:
        async def list_tools(self, *, cursor=None, cache_mode="use"):
            await asyncio.sleep(0.05)
            return ListToolsResult(tools=[], nextCursor=None)

    @asynccontextmanager
    async def immediate_open(config):
        yield DiscoveryClient()

    monkeypatch.setattr("mycode.mcp.manager.open_mcp_client", immediate_open)
    config = MCPConfig.model_validate({"mcpServers": {"local": {
        "transport": "stdio",
        "command": "local",
        "connect_timeout": 0.01,
        "tool_timeout": 0.2,
    }}})
    manager = MCPManager(config)

    manager.start()
    try:
        assert manager.statuses[0].status == "connected"
    finally:
        manager.close()


def test_shutdown_timeout_retains_live_thread_and_reports_state() -> None:
    class StubbornThread:
        def join(self, timeout=None):
            return None

        def is_alive(self):
            return True

    events = []
    manager = MCPManager(MCPConfig(), observability_sink=events.append)
    thread = StubbornThread()
    manager._thread = thread
    manager.shutdown_timeout = 0.01

    manager.close()

    assert manager._thread is thread
    assert manager.shutdown_status.status == "timeout"
    assert manager.shutdown_status.error == "shutdown_timeout"
    assert events[-1]["event_type"] == "mcp_shutdown"
    assert events[-1]["status"] == "timeout"
