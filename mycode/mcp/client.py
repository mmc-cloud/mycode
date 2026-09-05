from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager

import httpx2
from mcp import Client, StdioServerParameters
from mcp.client.streamable_http import streamable_http_client

from mycode.mcp.config import MCPServerConfig


@asynccontextmanager
async def open_mcp_client(config: MCPServerConfig) -> AsyncIterator[Client]:
    async with AsyncExitStack() as stack:
        if config.transport == "stdio":
            client = Client(
                StdioServerParameters(
                    command=config.command,
                    args=config.args,
                    env=config.env or None,
                ),
                read_timeout_seconds=config.tool_timeout,
            )
        else:
            http_client = await stack.enter_async_context(
                httpx2.AsyncClient(
                    headers=config.headers,
                    event_hooks={"response": [_raise_auth_status]},
                    follow_redirects=True,
                    timeout=config.tool_timeout,
                )
            )
            transport = streamable_http_client(
                config.url,
                http_client=http_client,
                terminate_on_close=False,
            )
            client = Client(transport, read_timeout_seconds=config.tool_timeout)
        yield await stack.enter_async_context(client)


async def _raise_auth_status(response: httpx2.Response) -> None:
    """Preserve 401/403 status without reading or exposing response content."""
    if response.status_code in (401, 403):
        response.raise_for_status()
