from collections.abc import Iterator

import pytest

from mycode.application.sessions import SessionStartRequest, start_project_session
from mycode.persistence.session_store import SessionStore
from mycode.presentation.cli.session_menu import select_session_request
from mycode.project import ProjectIdentity


def test_application_request_rejects_presentation_select_mode() -> None:
    with pytest.raises(ValueError, match="Unsupported session start mode"):
        SessionStartRequest(mode="select")


def test_empty_project_menu_requests_new_session(tmp_path) -> None:
    store = SessionStore(tmp_path / "projects")
    project = ProjectIdentity.from_workspace(tmp_path)
    output: list[str] = []

    request = select_session_request(store, project, output_func=output.append)

    assert request == SessionStartRequest(mode="new")
    assert output == [
        f"session> 项目 {tmp_path.resolve()}",
        "session> 没有历史会话，正在创建新会话",
    ]


def test_menu_invalid_input_then_resume_does_not_leak_other_project(
    tmp_path,
) -> None:
    store = SessionStore(tmp_path / "projects")
    project = ProjectIdentity.from_workspace(tmp_path)
    other_path = tmp_path / "other"
    other_path.mkdir()
    other = ProjectIdentity.from_workspace(other_path)
    store.create_session(other, title="secret", session_id="other")
    current = store.create_session(project, title="current", session_id="current")
    answers: Iterator[str] = iter(["invalid", "1"])
    output: list[str] = []

    request = select_session_request(
        store,
        project,
        input_func=lambda _prompt: next(answers),
        output_func=output.append,
    )

    assert request == SessionStartRequest(mode="resume", session_id=current.id)
    assert not any("secret" in line for line in output)
    assert any("选择无效" in line for line in output)


@pytest.mark.parametrize("answer", ["n", "new"])
def test_menu_new_option_returns_explicit_request(tmp_path, answer) -> None:
    store = SessionStore(tmp_path / "projects")
    project = ProjectIdentity.from_workspace(tmp_path)
    store.create_session(project, session_id="existing")

    request = select_session_request(
        store,
        project,
        input_func=lambda _prompt: answer,
        output_func=lambda _message: None,
    )

    assert request == SessionStartRequest(mode="new")


@pytest.mark.parametrize("confirm,deleted", [("DELETE one", True), ("wrong", False)])
def test_menu_delete_requires_exact_confirmation(
    tmp_path,
    confirm,
    deleted,
) -> None:
    store = SessionStore(tmp_path / "projects")
    project = ProjectIdentity.from_workspace(tmp_path)
    store.create_session(project, session_id="one")
    answers: Iterator[str] = iter(["d", "1", confirm, "q"])

    request = select_session_request(
        store,
        project,
        input_func=lambda _prompt: next(answers),
        output_func=lambda _message: None,
    )

    assert request is None
    assert (store.get_session(project, "one") is None) == deleted


def test_empty_menu_rejects_delete_option_after_last_session_is_removed(
    tmp_path,
) -> None:
    store = SessionStore(tmp_path / "projects")
    project = ProjectIdentity.from_workspace(tmp_path)
    store.create_session(project, session_id="one")
    answers: Iterator[str] = iter(["d", "1", "DELETE one", "d", "q"])
    output: list[str] = []

    request = select_session_request(
        store,
        project,
        input_func=lambda _prompt: next(answers),
        output_func=output.append,
    )

    assert request is None
    assert output.count("session> 请选择要永久删除的会话") == 1
    assert any("选择无效" in line for line in output)


def test_menu_eof_returns_none(tmp_path) -> None:
    store = SessionStore(tmp_path / "projects")
    project = ProjectIdentity.from_workspace(tmp_path)
    store.create_session(project, session_id="one")
    output: list[str] = []

    def eof(_prompt: str) -> str:
        raise EOFError

    assert select_session_request(
        store,
        project,
        input_func=eof,
        output_func=output.append,
    ) is None
    assert output[-2] == "session> [Q] 退出"
    assert output[-1] == ""


def test_menu_returns_resume_request_without_stale_active_precheck(tmp_path) -> None:
    store = SessionStore(tmp_path / "projects")
    project = ProjectIdentity.from_workspace(tmp_path)
    active = start_project_session(
        store,
        project,
        request=SessionStartRequest(mode="new"),
    )
    resumed = None

    def release_then_select(_prompt: str) -> str:
        active.interrupt()
        return "1"

    output: list[str] = []
    try:
        request = select_session_request(
            store,
            project,
            input_func=release_then_select,
            output_func=output.append,
        )
        assert request == SessionStartRequest(mode="resume", session_id=active.record.id)
        resumed = start_project_session(store, project, request=request)
    finally:
        active.interrupt()
        if resumed is not None:
            resumed.interrupt()

    assert not any("当前不可用" in line for line in output)
