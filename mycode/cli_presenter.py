from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

from mycode.agent import AgentEvent, AgentToolCall
from mycode.event_format import summarize_event_content, summarize_tool_arguments


CliDisplayMode = Literal["normal", "verbose", "debug"]


READ_ONLY_ACTIVITY_TOOLS = {
    "glob",
    "grep",
    "inspect_changes",
    "list_memories",
    "read_artifact",
    "read_file",
}


@dataclass
class CliPresenter:
    output: Callable[[str], None]
    mode: CliDisplayMode = "normal"
    _read_only_calls: list[AgentToolCall] = field(default_factory=list, init=False)
    _pending_result_tools: deque[str] = field(default_factory=deque, init=False)
    _continue_after_error: bool = field(default=False, init=False)

    def flush(self) -> None:
        if not self._read_only_calls:
            return
        self.output(_summarize_read_only_activity(self._read_only_calls))
        self._read_only_calls.clear()

    def show_agent_event(self, event: AgentEvent) -> None:
        if event.type == "turn":
            if event.turn_number is None or event.max_turns is None:
                return
            label = "轮次" if self.mode == "normal" else "turn"
            self.output(f"{label}> {event.turn_number}/{event.max_turns}")
            if event.content:
                notice_label = "提醒" if self.mode == "normal" else "turn_notice"
                self.output(f"{notice_label}> {event.content}")
            return

        if event.type == "context":
            if self.mode != "normal":
                self.output(f"context> {event.content}")
            return

        if event.type == "progress":
            if self.mode == "debug" and event.progress is not None:
                progress = event.progress
                self.output(
                    "progress> "
                    f"stagnation_turns={progress.stagnation_turns} "
                    f"same_tool_repeat={progress.same_tool_repeat} "
                    f"same_result_repeat={progress.same_result_repeat} "
                    f"resource_repeat={progress.resource_repeat} "
                    f"convergence_guided={progress.convergence_guided} "
                    f"reason={progress.reason}"
                )
            return

        if event.type == "artifact_warning":
            self.flush()
            if self.mode == "normal":
                self.output(f"警告> 工具结果归档异常：{event.content}")
            else:
                self.output(f"artifact> warning: {event.content}")
            return

        if event.type == "model_retry" and event.model_retry is not None:
            self.flush()
            retry = event.model_retry
            if self.mode == "normal":
                self.output(
                    "重试> 模型连接异常，"
                    f"{retry.delay_seconds:.1f} 秒后重试当前模型轮次"
                    f"（{retry.attempt}/{retry.max_retries}）"
                )
            else:
                self.output(
                    "model_retry> "
                    f"call_kind={retry.call_kind} "
                    f"attempt={retry.attempt}/{retry.max_retries} "
                    f"error_type={retry.error_type} "
                    f"error_code={retry.error_code} "
                    f"retryable={retry.retryable} "
                    f"delay_seconds={retry.delay_seconds:.1f} "
                    f"stream_started={retry.stream_started} "
                    f"partial_output_chars={retry.partial_output_chars}"
                )
            return

        if event.type == "tool_call" and event.tool_call is not None:
            self._pending_result_tools.append(event.tool_call.name)
            if self.mode == "normal":
                if self._continue_after_error:
                    self.flush()
                    self.output("活动> 上一步失败，正在尝试其他方式")
                    self._continue_after_error = False
                if event.tool_call.name in READ_ONLY_ACTIVITY_TOOLS:
                    self._read_only_calls.append(event.tool_call)
                    return
                self.flush()
                if event.tool_call.name != "delegate_task":
                    self.output(_describe_activity(event.tool_call))
                return
            argument_summary = summarize_tool_arguments(
                event.tool_call.name,
                event.tool_call.arguments,
            )
            self.output(
                f"tool_call> {event.tool_call.name}"
                f"({argument_summary})"
            )
            return

        if event.type == "tool_result" and event.tool_result is not None:
            self.flush()
            result = event.tool_result
            status = "ok" if result.ok else "error"
            tool_name = (
                self._pending_result_tools.popleft()
                if self._pending_result_tools
                else str(result.metadata.get("tool_name", "工具"))
            )
            if result.metadata.get("tool_name") == "delegate_task":
                metadata = result.metadata
                child_status = metadata.get("child_status", status)
                if self.mode == "normal" and result.ok and child_status == "completed":
                    return
                if self.mode == "normal":
                    reason = metadata.get(
                        "child_stop_reason",
                        metadata.get("reason", "未知原因"),
                    )
                    self.output(
                        f"警告> SubAgent 未完成：角色={metadata.get('role', 'unknown')}，"
                        f"状态={child_status}，原因={reason}"
                    )
                    self._continue_after_error = True
                    return
                details = [
                    f"role={metadata.get('role', 'unknown')}",
                    f"status={child_status}",
                    "stop="
                    f"{metadata.get('child_stop_reason', metadata.get('reason', 'unknown'))}",
                ]
                self.output(f"tool_result> {status}: subagent " + " ".join(details))
                return
            if self.mode == "normal" and result.ok:
                return
            summary = summarize_event_content(
                result.content if result.ok else result.error
            )
            if self.mode == "normal":
                self.output(
                    f"警告> {_tool_display_name(tool_name)}失败："
                    f"{_localize_error(summary)}"
                )
                self._continue_after_error = True
                return
            self.output(f"tool_result> {status}: {summary}")
            return

        if event.type == "error":
            self.flush()
            message = _localize_error(event.error or event.content)
            if self.mode == "normal":
                self.output(f"错误> {message}")
            else:
                self.output(f"error> {message}")
            return

        if event.type == "stop":
            self.flush()
            if event.stop_reason == "max_turns":
                message = event.content or (
                    "本轮已达到步骤上限，未能生成最终答案。"
                )
                self.output(f"警告> {message}")
                self.output(
                    "提示> 当前进度已保存；如需继续，直接输入“继续”。"
                )
                if self.mode == "debug":
                    self.output("stop> max_turns")
                return
            if self.mode == "normal":
                stop_message = _normal_stop_message(event.stop_reason)
                if stop_message is not None:
                    self.output(stop_message)
                return
            if self.mode == "debug" or event.stop_reason != "final_answer":
                self.output(f"stop> {event.stop_reason}")


def _summarize_read_only_activity(calls: list[AgentToolCall]) -> str:
    if len(calls) == 1:
        return _describe_activity(calls[0])
    if all(call.name == "read_file" for call in calls):
        targets = [_read_file_target(call) for call in calls]
        return f"活动> 正在读取 {len(targets)} 个文件：{', '.join(targets)}"
    if all(call.name == "read_artifact" for call in calls):
        return f"活动> 正在读取 {len(calls)} 份已保存的工具结果"
    if all(call.name == "grep" for call in calls):
        return f"活动> 正在执行 {len(calls)} 次代码搜索"
    counts = {name: 0 for name in READ_ONLY_ACTIVITY_TOOLS}
    for call in calls:
        counts[call.name] += 1
    summaries = []
    labels = {
        "read_file": "读取 {count} 个文件",
        "grep": "搜索代码 {count} 次",
        "glob": "查找文件 {count} 次",
        "read_artifact": "读取 {count} 份已保存的工具结果",
        "inspect_changes": "检查代码变更 {count} 次",
        "list_memories": "读取长期记忆 {count} 次",
    }
    for name in (
        "read_file",
        "grep",
        "glob",
        "read_artifact",
        "inspect_changes",
        "list_memories",
    ):
        if counts[name] > 0:
            summaries.append(labels[name].format(count=counts[name]))
    return "活动> 正在进行只读分析：" + "、".join(summaries)


def _describe_activity(call: AgentToolCall) -> str:
    arguments = call.arguments
    if call.name == "read_file":
        path = arguments.get("path", "未知文件")
        start_line = arguments.get("start_line")
        if isinstance(start_line, int) and start_line > 1:
            return f"活动> 正在继续读取 {path}（从第 {start_line} 行开始）"
        return f"活动> 正在读取文件：{path}"
    if call.name == "read_artifact":
        offset = arguments.get("offset_chars")
        if isinstance(offset, int) and offset > 0:
            return (
                "活动> 正在继续读取已保存的工具结果"
                f"（从第 {offset} 个字符开始）"
            )
        return "活动> 正在读取已保存的工具结果"
    if call.name == "glob":
        return f"活动> 正在查找文件：{arguments.get('pattern', '未知范围')}"
    if call.name == "grep":
        return (
            "活动> 正在搜索代码："
            f"{arguments.get('path_pattern', '工作区内的文件')}"
        )
    if call.name == "inspect_changes":
        return "活动> 正在检查代码变更"
    if call.name == "list_memories":
        return "活动> 正在读取长期记忆"
    activity_names = {
        "delete_memory": "正在删除长期记忆",
        "edit_file": "正在编辑文件",
        "run_command": "正在运行命令",
        "save_memory": "正在保存长期记忆",
        "write_file": "正在写入文件",
    }
    description = activity_names.get(call.name, f"正在执行 {call.name}")
    return f"活动> {description}"


def _read_file_target(call: AgentToolCall) -> str:
    path = str(call.arguments.get("path", "未知文件"))
    start_line = call.arguments.get("start_line")
    if isinstance(start_line, int) and start_line > 1:
        return f"{path}（从第 {start_line} 行继续）"
    return path


def _tool_display_name(name: str) -> str:
    names = {
        "glob": "查找文件",
        "grep": "搜索代码",
        "read_artifact": "读取已保存的工具结果",
        "read_file": "读取文件",
    }
    return names.get(name, name)


def _localize_error(message: str) -> str:
    translations = {
        "File not found: ": "未找到文件：",
        "Directory not found: ": "未找到目录：",
        "Model streaming failed: ": "模型流式请求失败：",
        "Model request failed: ": "模型请求失败：",
    }
    for prefix, translated in translations.items():
        if message.startswith(prefix):
            return translated + message[len(prefix) :]
    return message


def _normal_stop_message(stop_reason: str | None) -> str | None:
    messages = {
        "context_overflow": "错误> 当前上下文超过模型可用范围，本轮无法继续。",
        "model_error": "提示> 本轮因模型调用错误而结束。",
        "repeated_tool_call": "警告> 模型连续重复了相同的工具调用，本轮已停止。",
        "tool_error": "警告> 本轮因工具执行错误而停止。",
    }
    return messages.get(stop_reason)


__all__ = [
    "CliDisplayMode",
    "CliPresenter",
    "summarize_event_content",
    "summarize_tool_arguments",
]
