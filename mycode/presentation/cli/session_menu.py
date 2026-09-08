from __future__ import annotations

from collections.abc import Callable

from mycode.application.sessions import (
    SessionStartRequest,
    delete_project_session,
    list_project_sessions,
)
from mycode.persistence.session_store import (
    SessionRecord,
    SessionStore,
    SessionStoreError,
)
from mycode.project import ProjectIdentity


def select_session_request(
    store: SessionStore,
    project: ProjectIdentity,
    *,
    input_func: Callable[[str], str] = input,
    output_func: Callable[[str], None] = print,
) -> SessionStartRequest | None:
    sessions = list_project_sessions(store, project)
    output_func(f"session> 项目 {project.workspace_root}")
    if not sessions:
        output_func("session> 没有历史会话，正在创建新会话")
        return SessionStartRequest(mode="new")

    while True:
        if sessions:
            output_func("session> 请选择历史会话，或管理已有会话")
            _output_numbered_sessions(sessions, output_func)
        else:
            output_func("session> 当前没有历史会话")
        output_func("session> [N] 创建新会话")
        if sessions:
            output_func("session> [D] 永久删除会话")
        output_func("session> [Q] 退出")

        try:
            answer = input_func("session> ").strip()
        except EOFError:
            output_func("")
            return None
        normalized = answer.casefold()
        if normalized in {"n", "new"}:
            return SessionStartRequest(mode="new")
        if normalized in {"d", "delete"} and sessions:
            _delete_session_interactively(
                store,
                project,
                sessions,
                input_func=input_func,
                output_func=output_func,
            )
            sessions = list_project_sessions(store, project)
            continue
        if normalized in {"q", "quit"}:
            return None
        if answer.isdigit():
            index = int(answer)
            if 1 <= index <= len(sessions):
                selected = sessions[index - 1]
                return SessionStartRequest(mode="resume", session_id=selected.id)
        output_func("session> 选择无效，请重新输入")


def _output_numbered_sessions(
    sessions: list[SessionRecord],
    output_func: Callable[[str], None],
) -> None:
    for index, session in enumerate(sessions, start=1):
        output_func(
            f"session> [{index}] {session.title} "
            f"({_session_status_label(session.status)}, {session.id[:8]})"
        )


def _delete_session_interactively(
    store: SessionStore,
    project: ProjectIdentity,
    sessions: list[SessionRecord],
    *,
    input_func: Callable[[str], str],
    output_func: Callable[[str], None],
) -> None:
    output_func("session> 请选择要永久删除的会话")
    _output_numbered_sessions(sessions, output_func)
    output_func("session> [Q] 取消删除")

    target: SessionRecord | None = None
    while target is None:
        try:
            answer = input_func("session delete> ").strip()
        except EOFError:
            output_func("")
            output_func("session> 已取消删除")
            return
        if answer.casefold() in {"q", "quit", "cancel"}:
            output_func("session> 已取消删除")
            return
        if answer.isdigit():
            index = int(answer)
            if 1 <= index <= len(sessions):
                target = sessions[index - 1]
                break
        output_func("session> 删除选项无效，请重新输入")

    confirmation = f"DELETE {target.id}"
    output_func(
        f"session> 将永久删除 {target.title} "
        f"({_session_status_label(target.status)}, {target.id})"
    )
    output_func(
        "session> 这会删除该会话的消息、工具调用、Compact 状态、"
        "SubAgent 记录和 artifact"
    )
    try:
        answer = input_func(f"session> 输入 {confirmation} 进行确认：").strip()
    except EOFError:
        output_func("")
        output_func("session> 已取消删除")
        return
    if answer != confirmation:
        output_func("session> 已取消删除：确认文本不匹配")
        return

    try:
        removed = delete_project_session(store, project, target.id)
    except SessionStoreError as error:
        output_func(f"session> 当前无法删除：{error}")
        return
    if removed:
        output_func(f"session> 已永久删除 {target.id}：{target.title}")
    else:
        output_func(f"session> 会话此前已经删除：{target.id}")


def _session_status_label(status: str) -> str:
    return {
        "active": "进行中",
        "closed": "已关闭",
        "interrupted": "已中断",
    }.get(status, status)
