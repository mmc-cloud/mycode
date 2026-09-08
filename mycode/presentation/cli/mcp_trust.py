"""Terminal presentation for the application MCP trust contract."""

from __future__ import annotations

from collections.abc import Callable

from mycode.mcp.trust import MCPTrustRequest, MCPTrustWarning


class TerminalMCPTrustConfirmer:
    def __init__(
        self,
        input_func: Callable[[str], str] = input,
        output_func: Callable[[str], None] = print,
    ) -> None:
        self.input_func = input_func
        self.output_func = output_func

    def confirm(self, request: MCPTrustRequest) -> bool:
        self._show_request(request)
        try:
            approved = self.input_func(
                "MCP trust> 是否信任并启用这些项目级 MCP？[y/N] "
            ).strip().lower() in {"y", "yes"}
        except EOFError:
            approved = False

        if not approved:
            self.output_func("MCP trust> 未启用项目级 MCP")
        return approved

    def report_warning(self, warning: MCPTrustWarning) -> None:
        if warning.code == "invalid_trust_store":
            self.output_func(
                "MCP trust> 警告：信任状态文件无效；"
                "项目级 MCP 将按未信任处理"
            )
        elif warning.code == "persistence_failed":
            self.output_func(
                "MCP trust> 警告：信任状态未能保存；"
                "本次仍会启用，下次将再次询问"
            )

    def _show_request(self, request: MCPTrustRequest) -> None:
        self.output_func("MCP trust> 项目配置请求启用以下 MCP Server：")
        for server in request.servers:
            self.output_func("")
            self.output_func(f"server> {server.alias!r}")
            self.output_func(f"transport> {server.transport}")
            if server.transport == "stdio":
                self.output_func(f"command> {server.command!r}")
                self.output_func(f"args> {list(server.args)!r}")
                if server.env_keys:
                    self.output_func(f"env keys> {list(server.env_keys)!r}")
                continue

            self.output_func(f"url template> {server.url_template!r}")
            if server.destination is not None:
                self.output_func(f"destination> {server.destination!r}")
            if server.header_keys:
                self.output_func(f"header keys> {list(server.header_keys)!r}")
