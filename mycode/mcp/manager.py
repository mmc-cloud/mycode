import asyncio
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
import random
from threading import Event, Thread
from time import monotonic

from mcp import Client
from mcp.types import CallToolResult

from mycode.mcp.client import open_mcp_client
from mycode.mcp.config import (
    DEFAULT_MCP_SHUTDOWN_TIMEOUT_SECONDS,
    MCPConfig,
    MCPServerConfig,
)
from mycode.mcp.errors import (
    classify_mcp_error,
    is_transient_mcp_error,
    safe_error_summary,
)
from mycode.mcp.models import MCPServerStatus, MCPShutdownStatus
from mycode.mcp.tool_adapter import MCPToolAdapter
from mycode.observability import ObservationSink, emit_observation


MCP_STARTUP_RETRY_MIN_DELAY_SECONDS = 1.0
MCP_STARTUP_RETRY_MAX_DELAY_SECONDS = 2.0
MCP_STARTUP_WAIT_SAFETY_MARGIN_SECONDS = 5.0


@dataclass
class MCPManager:
    config: MCPConfig
    observability_sink: ObservationSink | None = None
    shutdown_timeout: float = DEFAULT_MCP_SHUTDOWN_TIMEOUT_SECONDS
    statuses: tuple[MCPServerStatus, ...] = field(default=(), init=False)
    shutdown_status: MCPShutdownStatus = field(
        default_factory=MCPShutdownStatus, init=False
    )
    tools: tuple[MCPToolAdapter, ...] = field(default=(), init=False)
    _clients: dict[str, Client] = field(default_factory=dict, init=False, repr=False)
    _tool_timeouts: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _loop: asyncio.AbstractEventLoop | None = field(default=None, init=False, repr=False)
    _stop: asyncio.Event | None = field(default=None, init=False, repr=False)
    _ready: Event = field(default_factory=Event, init=False, repr=False)
    _thread: Thread | None = field(default=None, init=False, repr=False)
    _fatal_error: BaseException | None = field(default=None, init=False, repr=False)

    def start(self) -> None:
        if self._thread is not None or not self.config.mcp_servers:
            return
        self._ready = Event()
        self._fatal_error = None
        self.statuses = ()
        self.tools = ()
        self.shutdown_status = MCPShutdownStatus()
        self._thread = Thread(target=self._thread_main, name="mycode-mcp", daemon=True)
        self._thread.start()
        wait_seconds = _startup_wait_seconds(self.config)
        if not self._ready.wait(wait_seconds):
            error = TimeoutError()
            self.statuses = tuple(
                MCPServerStatus(
                    alias=alias,
                    status="failed",
                    error_type=type(error).__name__,
                    error_summary=safe_error_summary(error),
                )
                for alias in self.config.mcp_servers
            )
            self.close()
        elif self._fatal_error is not None:
            self.statuses = tuple(
                MCPServerStatus(
                    alias=alias,
                    status="failed",
                    error_type=type(self._fatal_error).__name__,
                    error_summary=safe_error_summary(self._fatal_error),
                )
                for alias in self.config.mcp_servers
            )

    def close(self) -> None:
        started = monotonic()
        loop, stop, thread = self._loop, self._stop, self._thread
        if thread is None:
            self.shutdown_status = MCPShutdownStatus(status="completed")
            return
        if loop is not None and stop is not None and thread.is_alive():
            try:
                loop.call_soon_threadsafe(stop.set)
            except RuntimeError:
                pass
        thread.join(timeout=self.shutdown_timeout)
        if thread.is_alive():
            self.shutdown_status = MCPShutdownStatus(
                status="timeout", error="shutdown_timeout"
            )
            emit_observation(
                self.observability_sink,
                "mcp_shutdown",
                {
                    "status": "timeout",
                    "duration_ms": round((monotonic() - started) * 1000),
                    "error_type": "shutdown_timeout",
                },
            )
            return

        self.shutdown_status = MCPShutdownStatus(status="completed")
        self._thread = None
        self._loop = None
        self._stop = None
        self._clients.clear()
        self._tool_timeouts.clear()
        self.tools = ()
        self._ready = Event()
        emit_observation(
            self.observability_sink,
            "mcp_shutdown",
            {
                "status": "completed",
                "duration_ms": round((monotonic() - started) * 1000),
                "error_type": None,
            },
        )

    async def call_tool(
        self, server_alias: str, remote_name: str, arguments: dict[str, object]
    ) -> CallToolResult:
        loop = self._loop
        if loop is None or server_alias not in self._clients:
            raise RuntimeError("MCP server is unavailable")
        future = asyncio.run_coroutine_threadsafe(
            self._call_on_runtime(server_alias, remote_name, arguments), loop
        )
        return await asyncio.wrap_future(future)

    def _thread_main(self) -> None:
        try:
            asyncio.run(self._serve())
        except BaseException as error:  # noqa: BLE001 - thread boundary must signal ready
            self._fatal_error = error
            self._ready.set()

    async def _serve(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._stop = asyncio.Event()
        loop = asyncio.get_running_loop()
        startups: list[asyncio.Future[_ServerStartup]] = []
        tasks: list[asyncio.Task[None]] = []
        for alias, server in self.config.mcp_servers.items():
            startup: asyncio.Future[_ServerStartup] = loop.create_future()
            startups.append(startup)
            tasks.append(
                asyncio.create_task(self._serve_server(alias, server, startup))
            )
        try:
            results = await asyncio.gather(*startups)
            self.statuses = tuple(result.status for result in results)
            self.tools = tuple(
                tool for result in results for tool in result.tools
            )
            self._ready.set()
            await self._stop.wait()
        finally:
            self._stop.set()
            await asyncio.gather(*tasks, return_exceptions=True)
            self._clients.clear()
            self._tool_timeouts.clear()

    async def _serve_server(
        self,
        alias: str,
        server: MCPServerConfig,
        startup: asyncio.Future["_ServerStartup"],
    ) -> None:
        started = monotonic()
        retry_count = 0
        for attempt in range(1, 3):
            try:
                async with AsyncExitStack() as stack:
                    async with asyncio.timeout(server.connect_timeout):
                        client = await stack.enter_async_context(
                            open_mcp_client(server)
                        )
                    async with asyncio.timeout(server.tool_timeout):
                        discovered = await _discover_all_tools(client)
                    remote_names = [tool.name for tool in discovered]
                    if len(remote_names) != len(set(remote_names)):
                        raise ValueError("MCP server returned duplicate tool names")
                    adapters = tuple(
                        MCPToolAdapter(alias, tool, self.call_tool)
                        for tool in discovered
                    )
                    self._clients[alias] = client
                    self._tool_timeouts[alias] = server.tool_timeout
                    result = _ServerStartup(
                        status=MCPServerStatus(
                            alias=alias,
                            status="connected",
                            tool_count=len(adapters),
                        ),
                        tools=adapters,
                    )
                    self._report_server_start(
                        startup,
                        result,
                        started,
                        attempt=attempt,
                        retry_count=retry_count,
                        recovered_after_retry=retry_count > 0,
                    )
                    await self._stop.wait()
                    return
            except Exception as error:  # noqa: BLE001 - isolate one external server
                classified = classify_mcp_error(error)
                if attempt == 1 and is_transient_mcp_error(classified):
                    retry_count = 1
                    await asyncio.sleep(
                        random.uniform(
                            MCP_STARTUP_RETRY_MIN_DELAY_SECONDS,
                            MCP_STARTUP_RETRY_MAX_DELAY_SECONDS,
                        )
                    )
                    continue
                result = _ServerStartup(
                    status=MCPServerStatus(
                        alias=alias,
                        status="failed",
                        error_type=type(error).__name__,
                        error_summary=classified.summary,
                    )
                )
                self._report_server_start(
                    startup,
                    result,
                    started,
                    attempt=attempt,
                    retry_count=retry_count,
                    recovered_after_retry=False,
                    error_category=classified.category,
                )
                return
            finally:
                self._clients.pop(alias, None)
                self._tool_timeouts.pop(alias, None)

    def _report_server_start(
        self,
        startup: asyncio.Future["_ServerStartup"],
        result: "_ServerStartup",
        started: float,
        *,
        attempt: int,
        retry_count: int,
        recovered_after_retry: bool,
        error_category: str | None = None,
    ) -> None:
        if not startup.done():
            startup.set_result(result)
        emit_observation(
            self.observability_sink,
            "mcp_server_start",
            {
                "server_alias": result.status.alias,
                "status": result.status.status,
                "tool_count": result.status.tool_count,
                "duration_ms": round((monotonic() - started) * 1000),
                "error_type": result.status.error_type,
                "error_category": error_category,
                "attempt": attempt,
                "retry_count": retry_count,
                "recovered_after_retry": recovered_after_retry,
            },
        )

    async def _call_on_runtime(
        self, alias: str, name: str, arguments: dict[str, object]
    ) -> CallToolResult:
        started = monotonic()
        status = "ok"
        error_type = None
        error_category = None
        root_error_type = None
        try:
            async with asyncio.timeout(self._tool_timeouts[alias]):
                return await self._clients[alias].call_tool(name, arguments)
        except Exception as error:
            status = "error"
            classified = classify_mcp_error(error)
            error_type = type(error).__name__
            error_category = classified.category
            root_error_type = classified.root_error_type
            raise
        finally:
            emit_observation(
                self.observability_sink,
                "mcp_tool_call",
                {
                    "server_alias": alias,
                    "tool_name": name,
                    "status": status,
                    "duration_ms": round((monotonic() - started) * 1000),
                    "error_type": error_type,
                    "root_error_type": root_error_type,
                    "error_category": error_category,
                },
            )


async def _discover_all_tools(client: Client) -> list[object]:
    tools: list[object] = []
    cursor: str | None = None
    while True:
        page = await client.list_tools(cursor=cursor, cache_mode="refresh")
        tools.extend(page.tools)
        cursor = page.next_cursor
        if cursor is None:
            return tools


@dataclass(frozen=True)
class _ServerStartup:
    status: MCPServerStatus
    tools: tuple[MCPToolAdapter, ...] = ()


def _startup_wait_seconds(config: MCPConfig) -> float:
    max_server_timeout = max(
        server.connect_timeout + server.tool_timeout
        for server in config.mcp_servers.values()
    )
    return (
        2 * max_server_timeout
        + MCP_STARTUP_RETRY_MAX_DELAY_SECONDS
        + MCP_STARTUP_WAIT_SAFETY_MARGIN_SECONDS
    )
