from collections.abc import Callable

from mycode.permissions import ConfirmationRequest, ConfirmationResult


class TerminalConfirmer:
    def __init__(
        self,
        input_func: Callable[[str], str] = input,
        output_func: Callable[[str], None] = print,
    ) -> None:
        self.input_func = input_func
        self.output_func = output_func

    def confirm(self, request: ConfirmationRequest) -> ConfirmationResult:
        permission_request = request.permission_request
        permission_decision = request.permission_decision

        self.output_func(
            f"permission> {permission_request.tool_name} 需要确认"
        )
        if permission_request.target is not None:
            self.output_func(f"target> {permission_request.target}")
        self._output_metadata("resolved_path", request)
        self._output_metadata("workspace_root", request)
        self._output_metadata("path_scope", request)
        self._output_metadata("pattern_scope", request)
        self._output_metadata("command_display", request)
        self._output_metadata("resolved_cwd", request)
        self._output_metadata("cwd_scope", request)
        self._output_metadata("command_risk_category", request)
        self._output_metadata("command_risk", request)
        self._output_metadata("command_risk_reason", request)
        self._output_metadata("memory_scope", request)
        self._output_metadata("memory_kind", request)
        self._output_metadata("memory_key", request)
        self._output_metadata("memory_content", request)
        self._output_metadata("memory_path", request)
        self.output_func(f"reason> {permission_decision.reason}")
        if request.prompt:
            self.output_func(
                f"message> {_localize_confirmation_message(request.prompt)}"
            )

        try:
            answer = self.input_func("是否批准？[y/N] ").strip().lower()
        except EOFError:
            return ConfirmationResult.rejected(
                message="Permission confirmation unavailable.",
                metadata={"input": "eof"},
            )

        if answer in {"y", "yes"}:
            return ConfirmationResult.approved(
                message="Permission confirmation approved.",
                metadata={"input": answer},
            )

        return ConfirmationResult.rejected(
            message="Permission confirmation rejected.",
            metadata={"input": answer},
        )

    def _output_metadata(
        self,
        key: str,
        request: ConfirmationRequest,
    ) -> None:
        value = _metadata_value(request, key)
        if value is not None:
            if key == "memory_content":
                self.output_func(f"{key}> {value!r}")
                return
            self.output_func(f"{key}> {value}")


def _metadata_value(
    request: ConfirmationRequest,
    key: str,
) -> object | None:
    if key in request.metadata:
        return request.metadata[key]

    return request.permission_decision.metadata.get(key)


def _localize_confirmation_message(message: str) -> str:
    exact_translations = {
        "Sensitive path requires confirmation.": "敏感路径需要确认。",
    }
    if message in exact_translations:
        return exact_translations[message]
    translations = {
        "Path outside workspace requires confirmation: ": "工作区外路径需要确认：",
        "Sensitive path requires confirmation: ": "敏感路径需要确认：",
        "Ignored path requires confirmation: ": "忽略路径需要确认：",
        "Sensitive path pattern requires confirmation: ": "敏感路径模式需要确认：",
        "Write operation requires confirmation: ": "写操作需要确认：",
        "Command operation requires confirmation: ": "命令操作需要确认：",
    }
    for prefix, translated in translations.items():
        if message.startswith(prefix):
            return translated + message[len(prefix) :]
    return message
