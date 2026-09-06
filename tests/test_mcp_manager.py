import asyncio
from contextlib import asynccontextmanager
from time import monotonic

import httpx2
from mcp.types import (
    CallToolResult,
    ListToolsResult,
    TextContent,
    Tool,
    ToolAnnotations,
)

from mycode.mcp.config import MCPConfig
from mycode.mcp.manager import (
    MCP_STARTUP_RETRY_MAX_DELAY_SECONDS,
    MCP_STARTUP_WAIT_SAFETY_MARGIN_SECONDS,
    MCPManager,
    _startup_wait_seconds,
)


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
    assert sorted(closed) == ["broken", "broken", "good"]
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


def test_startup_wait_includes_retry_delay_and_safety_margin() -> None:
    config = MCPConfig.model_validate({"mcpServers": {
        "remote": {
            "transport": "stdio",
            "command": "remote",
            "connect_timeout": 0.5,
            "tool_timeout": 1.0,
        },
    }})

    assert _startup_wait_seconds(config) == (
        2 * 1.5
        + MCP_STARTUP_RETRY_MAX_DELAY_SECONDS
        + MCP_STARTUP_WAIT_SAFETY_MARGIN_SECONDS
    )


def test_transient_startup_failure_retries_once_and_recovers(monkeypatch) -> None:
    attempts = 0

    class RetryingClient:
        async def list_tools(self, *, cursor=None, cache_mode="use"):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise httpx2.ConnectError("secret transport")
            return ListToolsResult(tools=[], nextCursor=None)

    @asynccontextmanager
    async def fake_open(config):
        yield RetryingClient()

    async def no_sleep(delay):
        assert delay == 1.25

    monkeypatch.setattr("mycode.mcp.manager.open_mcp_client", fake_open)
    monkeypatch.setattr("mycode.mcp.manager.random.uniform", lambda _a, _b: 1.25)
    monkeypatch.setattr("mycode.mcp.manager.asyncio.sleep", no_sleep)
    config = MCPConfig.model_validate(
        {"mcpServers": {"remote": {"transport": "stdio", "command": "remote"}}}
    )
    events = []
    manager = MCPManager(config, observability_sink=events.append)

    manager.start()
    try:
        assert attempts == 2
        assert manager.statuses[0].status == "connected"
        event = next(
            item
            for item in events
            if item["event_type"] == "mcp_server_start"
        )
        assert event["attempt"] == 2
        assert event["retry_count"] == 1
        assert event["recovered_after_retry"] is True
        assert event["error_category"] is None
    finally:
        manager.close()


def test_two_transient_startup_failures_end_failed(monkeypatch) -> None:
    attempts = 0

    class FailingClient:
        async def list_tools(self, *, cursor=None, cache_mode="use"):
            nonlocal attempts
            attempts += 1
            raise httpx2.ConnectError("secret transport")

    @asynccontextmanager
    async def fake_open(config):
        yield FailingClient()

    async def no_sleep(delay):
        return None

    monkeypatch.setattr("mycode.mcp.manager.open_mcp_client", fake_open)
    monkeypatch.setattr("mycode.mcp.manager.random.uniform", lambda _a, _b: 1.0)
    monkeypatch.setattr("mycode.mcp.manager.asyncio.sleep", no_sleep)
    config = MCPConfig.model_validate(
        {"mcpServers": {"remote": {"transport": "stdio", "command": "remote"}}}
    )
    events = []
    manager = MCPManager(config, observability_sink=events.append)

    manager.start()
    try:
        assert attempts == 2
        assert manager.statuses[0].status == "failed"
        assert manager.statuses[0].error_summary == "Connection failed"
        event = next(
            item
            for item in events
            if item["event_type"] == "mcp_server_start"
        )
        assert event["attempt"] == 2
        assert event["retry_count"] == 1
        assert event["recovered_after_retry"] is False
        assert event["error_category"] == "connection_error"
    finally:
        manager.close()


def test_authentication_failure_does_not_retry_startup(monkeypatch) -> None:
    attempts = 0
    request = httpx2.Request("GET", "https://example.test/mcp")
    response = httpx2.Response(401, request=request)

    class AuthFailingClient:
        async def list_tools(self, *, cursor=None, cache_mode="use"):
            nonlocal attempts
            attempts += 1
            raise httpx2.HTTPStatusError(
                "Bearer secret-token", request=request, response=response
            )

    @asynccontextmanager
    async def fake_open(config):
        yield AuthFailingClient()

    monkeypatch.setattr("mycode.mcp.manager.open_mcp_client", fake_open)
    config = MCPConfig.model_validate(
        {"mcpServers": {"remote": {"transport": "stdio", "command": "remote"}}}
    )
    manager = MCPManager(config)

    manager.start()
    try:
        assert attempts == 1
        assert manager.statuses[0].error_summary == "Authentication failed (401)"
    finally:
        manager.close()


def test_permission_error_does_not_retry_startup(monkeypatch) -> None:
    attempts = 0

    class PermissionFailingClient:
        async def list_tools(self, *, cursor=None, cache_mode="use"):
            nonlocal attempts
            attempts += 1
            raise PermissionError("private server path")

    @asynccontextmanager
    async def fake_open(config):
        yield PermissionFailingClient()

    monkeypatch.setattr("mycode.mcp.manager.open_mcp_client", fake_open)
    config = MCPConfig.model_validate(
        {"mcpServers": {"remote": {"transport": "stdio", "command": "remote"}}}
    )
    manager = MCPManager(config)

    manager.start()
    try:
        assert attempts == 1
        assert manager.statuses[0].error_summary == "Permission denied"
    finally:
        manager.close()


def test_generic_oserror_does_not_retry_startup(monkeypatch) -> None:
    attempts = 0

    class ProcessFailingClient:
        async def list_tools(self, *, cursor=None, cache_mode="use"):
            nonlocal attempts
            attempts += 1
            raise OSError("private process detail")

    @asynccontextmanager
    async def fake_open(config):
        yield ProcessFailingClient()

    monkeypatch.setattr("mycode.mcp.manager.open_mcp_client", fake_open)
    config = MCPConfig.model_validate(
        {"mcpServers": {"remote": {"transport": "stdio", "command": "remote"}}}
    )
    manager = MCPManager(config)

    manager.start()
    try:
        assert attempts == 1
        assert manager.statuses[0].error_summary == "Server process or transport failed"
    finally:
        manager.close()


def test_duplicate_tool_schema_failure_does_not_retry_startup(monkeypatch) -> None:
    opens = 0

    class InvalidClient:
        async def list_tools(self, *, cursor=None, cache_mode="use"):
            return ListToolsResult(
                tools=[Tool(name="same", inputSchema={}), Tool(name="same", inputSchema={})],
                nextCursor=None,
            )

    @asynccontextmanager
    async def fake_open(config):
        nonlocal opens
        opens += 1
        yield InvalidClient()

    monkeypatch.setattr("mycode.mcp.manager.open_mcp_client", fake_open)
    config = MCPConfig.model_validate(
        {"mcpServers": {"remote": {"transport": "stdio", "command": "remote"}}}
    )
    manager = MCPManager(config)

    manager.start()
    try:
        assert opens == 1
        assert manager.statuses[0].status == "failed"
        assert manager.statuses[0].error_type == "ValueError"
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
