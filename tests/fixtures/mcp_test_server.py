import asyncio
import sys

import uvicorn
from mcp.server import MCPServer
from mcp.types import ToolAnnotations

server = MCPServer("mycode-test")
readonly = ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=False)


@server.tool(annotations=readonly)
def echo(text: str) -> str:
    return text


@server.tool(annotations=readonly, structured_output=True)
def add(a: int, b: int) -> dict[str, int]:
    return {"sum": a + b}


@server.tool(annotations=readonly)
def fail() -> str:
    raise ValueError("expected failure")


@server.tool(annotations=readonly)
async def slow() -> str:
    await asyncio.sleep(1)
    return "late"


if __name__ == "__main__":
    transport = "stdio" if len(sys.argv) == 1 else sys.argv[1]
    if transport == "stdio":
        server.run("stdio")
    elif transport == "streamable-http":
        port = int(sys.argv[2])

        class HeaderMiddleware:
            def __init__(self, app) -> None:
                self.app = app

            async def __call__(self, scope, receive, send) -> None:
                if scope["type"] == "http":
                    headers = dict(scope.get("headers", []))
                    if headers.get(b"x-mycode-test") != b"secret":
                        await send(
                            {
                                "type": "http.response.start",
                                "status": 401,
                                "headers": [
                                    (b"content-type", b"text/plain; charset=utf-8")
                                ],
                            }
                        )
                        await send(
                            {
                                "type": "http.response.body",
                                "body": b"missing test header",
                            }
                        )
                        return
                await self.app(scope, receive, send)

        app = HeaderMiddleware(
            server.streamable_http_app(
                streamable_http_path="/mcp",
                stateless_http=True,
                host="127.0.0.1",
            )
        )
        uvicorn.run(
            app,
            host="127.0.0.1",
            port=port,
            log_level="warning",
        )
    else:
        raise ValueError(f"unsupported transport: {transport}")
