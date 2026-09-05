import hashlib
import json
from pathlib import Path

import pytest

from mycode.mcp.config import load_mcp_config_layers
from mycode.mcp.manager import MCPManager
from mycode.mcp.trust import (
    apply_project_mcp_trust,
    project_mcp_fingerprint,
)
from mycode.project import ProjectIdentity


def write_mcp(path: Path, servers: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"mcpServers": servers}), encoding="utf-8")


def stdio(
    command: str = "python",
    *,
    args: list[str] | None = None,
    env: dict[str, str] | None = None,
) -> dict[str, object]:
    return {
        "transport": "stdio",
        "command": command,
        "args": [] if args is None else args,
        "env": {} if env is None else env,
    }


def http(
    url: str = "https://example.test/mcp",
    *,
    headers: dict[str, str] | None = None,
) -> dict[str, object]:
    return {
        "transport": "streamable_http",
        "url": url,
        "headers": {} if headers is None else headers,
    }


def load_layers(
    tmp_path: Path,
    *,
    user: dict[str, object] | None = None,
    project: dict[str, object] | None = None,
    environ: dict[str, str] | None = None,
    user_env: str | None = None,
):
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    user_path = tmp_path / "user.json"
    if user is not None:
        write_mcp(user_path, user)
    if project is not None:
        write_mcp(workspace / ".mycode" / "mcp.json", project)
    user_env_path = tmp_path / "user.env"
    if user_env is not None:
        user_env_path.write_text(user_env, encoding="utf-8")
    loaded = load_mcp_config_layers(
        user_path,
        env_file=user_env_path,
        environ={} if environ is None else environ,
        workspace_root=workspace,
    )
    return workspace, loaded


def decide(
    tmp_path: Path,
    loaded,
    answer: str = "n",
    *,
    input_func=None,
    outputs: list[str] | None = None,
):
    workspace = tmp_path / "workspace"
    project = ProjectIdentity.from_workspace(workspace)
    sink = [] if outputs is None else outputs
    return apply_project_mcp_trust(
        loaded,
        project,
        input_func=(lambda prompt: answer) if input_func is None else input_func,
        output_func=sink.append,
        trust_file=tmp_path / "mcp-trust.json",
    )


def _legacy_stdio_fingerprint(loaded) -> str:
    entries = []
    for alias in sorted(loaded.project.mcp_servers):
        resolved = loaded.project.mcp_servers[alias]
        if resolved.transport != "stdio":
            continue
        unresolved = loaded.project_unresolved.mcp_servers[alias]
        entries.append(
            {
                "alias": alias,
                "transport": "stdio",
                "command": {
                    "unresolved": unresolved.command,
                    "resolved": resolved.command,
                },
                "args": {
                    "unresolved": unresolved.args,
                    "resolved": resolved.args,
                },
                "env": unresolved.env,
                "connect_timeout": resolved.connect_timeout,
                "tool_timeout": resolved.tool_timeout,
            }
        )
    canonical = json.dumps(
        {"version": 1, "servers": entries},
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def test_user_stdio_does_not_require_project_trust(tmp_path) -> None:
    _, loaded = load_layers(tmp_path, user={"user": stdio("user")})

    config = decide(
        tmp_path,
        loaded,
        input_func=lambda prompt: pytest.fail("must not prompt"),
    )

    assert config.mcp_servers["user"].command == "user"


def test_user_http_does_not_require_project_trust(tmp_path) -> None:
    _, loaded = load_layers(tmp_path, user={"remote": http()})

    config = decide(
        tmp_path,
        loaded,
        input_func=lambda prompt: pytest.fail("must not prompt"),
    )

    assert set(config.mcp_servers) == {"remote"}


def test_project_http_first_use_requires_project_trust(tmp_path) -> None:
    _, loaded = load_layers(tmp_path, project={"remote": http()})
    prompts = []

    config = decide(
        tmp_path,
        loaded,
        input_func=lambda prompt: prompts.append(prompt) or "n",
    )

    assert config.mcp_servers == {}
    assert len(prompts) == 1


def test_rejected_project_http_never_calls_mcp_client_or_network_path(
    tmp_path, monkeypatch
) -> None:
    _, loaded = load_layers(tmp_path, project={"remote": http()})
    effective = decide(tmp_path, loaded, "n")
    client_calls = []

    def forbidden_client(config):
        client_calls.append(config)
        raise AssertionError("untrusted project HTTP reached client path")

    monkeypatch.setattr("mycode.mcp.manager.open_mcp_client", forbidden_client)
    manager = MCPManager(effective)

    manager.start()
    manager.close()

    assert effective.mcp_servers == {}
    assert client_calls == []


def test_approval_enables_project_http(tmp_path) -> None:
    _, loaded = load_layers(tmp_path, project={"remote": http()})

    config = decide(tmp_path, loaded, "yes")

    assert set(config.mcp_servers) == {"remote"}


@pytest.mark.parametrize("answer", ["y", "YES"])
def test_approval_enables_project_stdio_and_persists_trust(tmp_path, answer) -> None:
    workspace, loaded = load_layers(
        tmp_path,
        project={"local": stdio("python", args=["server.py"])},
    )

    config = decide(tmp_path, loaded, answer)
    payload = json.loads((tmp_path / "mcp-trust.json").read_text(encoding="utf-8"))

    assert set(config.mcp_servers) == {"local"}
    assert payload == {
        "projects": {
            ProjectIdentity.from_workspace(workspace).key: project_mcp_fingerprint(loaded)
        },
        "version": 1,
    }


@pytest.mark.parametrize("answer", ["", "n", "no", "maybe"])
def test_rejection_skips_all_project_mcp_but_keeps_user_mcp(
    tmp_path, answer
) -> None:
    _, loaded = load_layers(
        tmp_path,
        user={"user": stdio("user")},
        project={"local": stdio("project"), "remote": http()},
    )

    config = decide(tmp_path, loaded, answer)

    assert set(config.mcp_servers) == {"user"}
    assert not (tmp_path / "mcp-trust.json").exists()


def test_eof_rejects_project_stdio(tmp_path) -> None:
    _, loaded = load_layers(tmp_path, project={"local": stdio()})

    def raise_eof(prompt: str) -> str:
        raise EOFError

    config = decide(tmp_path, loaded, input_func=raise_eof)

    assert config.mcp_servers == {}


def test_multiple_project_stdio_servers_use_one_confirmation(tmp_path) -> None:
    _, loaded = load_layers(
        tmp_path,
        project={"one": stdio("one"), "two": stdio("two")},
    )
    prompts = []

    config = decide(
        tmp_path,
        loaded,
        input_func=lambda prompt: prompts.append(prompt) or "yes",
    )

    assert set(config.mcp_servers) == {"one", "two"}
    assert len(prompts) == 1


def test_project_stdio_and_http_use_one_confirmation(tmp_path) -> None:
    _, loaded = load_layers(
        tmp_path,
        project={"local": stdio(), "remote": http()},
    )
    prompts = []

    config = decide(
        tmp_path,
        loaded,
        input_func=lambda prompt: prompts.append(prompt) or "yes",
    )

    assert set(config.mcp_servers) == {"local", "remote"}
    assert len(prompts) == 1


def test_rejected_project_override_falls_back_to_user_alias(tmp_path) -> None:
    _, loaded = load_layers(
        tmp_path,
        user={"shared": stdio("trusted-user")},
        project={"shared": stdio("untrusted-project")},
    )

    config = decide(tmp_path, loaded)

    assert config.mcp_servers["shared"].command == "trusted-user"


def test_matching_trust_skips_prompt_on_next_run(tmp_path) -> None:
    _, loaded = load_layers(tmp_path, project={"local": stdio()})
    decide(tmp_path, loaded, "yes")

    config = decide(
        tmp_path,
        loaded,
        input_func=lambda prompt: pytest.fail("must not prompt"),
    )

    assert set(config.mcp_servers) == {"local"}


def test_matching_http_trust_skips_prompt_on_next_run(tmp_path) -> None:
    _, loaded = load_layers(tmp_path, project={"remote": http()})
    decide(tmp_path, loaded, "yes")

    config = decide(
        tmp_path,
        loaded,
        input_func=lambda prompt: pytest.fail("must not prompt"),
    )

    assert set(config.mcp_servers) == {"remote"}


@pytest.mark.parametrize(
    ("first", "changed"),
    [
        ({"a": stdio("python")}, {"a": stdio("python3")}),
        ({"a": stdio(args=["one"])}, {"a": stdio(args=["two"])}),
        ({"a": stdio()}, {"a": stdio(), "b": stdio()}),
        ({"a": stdio(), "b": stdio()}, {"a": stdio()}),
        ({"a": stdio(env={"TOKEN": "${ONE}"})}, {"a": stdio(env={"TOKEN": "${TWO}"})}),
    ],
)
def test_stdio_execution_config_changes_require_new_confirmation(
    tmp_path, first, changed
) -> None:
    _, original = load_layers(
        tmp_path,
        project=first,
        environ={"ONE": "secret", "TWO": "secret"},
    )
    decide(tmp_path, original, "yes")
    write_mcp(tmp_path / "workspace" / ".mycode" / "mcp.json", changed)
    _, updated = load_layers(
        tmp_path,
        project=changed,
        environ={"ONE": "secret", "TWO": "secret"},
    )
    prompts = []

    decide(
        tmp_path,
        updated,
        input_func=lambda prompt: prompts.append(prompt) or "n",
    )

    assert len(prompts) == 1


def test_resolved_command_and_args_changes_require_new_confirmation(tmp_path) -> None:
    project_config = {
        "local": stdio("${COMMAND}", args=["${ARG}"])
    }
    _, original = load_layers(
        tmp_path,
        project=project_config,
        environ={"COMMAND": "python", "ARG": "one"},
    )
    decide(tmp_path, original, "yes")
    _, updated = load_layers(
        tmp_path,
        project=project_config,
        environ={"COMMAND": "python3", "ARG": "two"},
    )
    prompts = []

    decide(
        tmp_path,
        updated,
        input_func=lambda prompt: prompts.append(prompt) or "n",
    )

    assert len(prompts) == 1


def test_unresolved_command_definition_change_invalidates_even_if_resolution_matches(
    tmp_path,
) -> None:
    first = {"local": stdio("${FIRST}", args=["${FIRST_ARG}"])}
    _, original = load_layers(
        tmp_path,
        project=first,
        environ={"FIRST": "python", "FIRST_ARG": "server.py"},
    )
    decide(tmp_path, original, "yes")
    changed = {"local": stdio("${SECOND}", args=["${SECOND_ARG}"])}
    write_mcp(tmp_path / "workspace" / ".mycode" / "mcp.json", changed)
    _, updated = load_layers(
        tmp_path,
        project=changed,
        environ={"SECOND": "python", "SECOND_ARG": "server.py"},
    )
    prompts = []

    decide(
        tmp_path,
        updated,
        input_func=lambda prompt: prompts.append(prompt) or "n",
    )

    assert len(prompts) == 1


@pytest.mark.parametrize("field", ["connect_timeout", "tool_timeout"])
def test_stdio_timeout_change_invalidates_trust(tmp_path, field) -> None:
    first_server = stdio()
    first_server[field] = 1
    _, original = load_layers(tmp_path, project={"local": first_server})
    decide(tmp_path, original, "yes")
    changed_server = stdio()
    changed_server[field] = 2
    changed = {"local": changed_server}
    write_mcp(tmp_path / "workspace" / ".mycode" / "mcp.json", changed)
    _, updated = load_layers(tmp_path, project=changed)
    prompts = []

    decide(
        tmp_path,
        updated,
        input_func=lambda prompt: prompts.append(prompt) or "n",
    )

    assert len(prompts) == 1


@pytest.mark.parametrize("field", ["connect_timeout", "tool_timeout"])
def test_http_timeout_change_invalidates_trust(tmp_path, field) -> None:
    first_server = http()
    first_server[field] = 1
    _, original = load_layers(tmp_path, project={"remote": first_server})
    decide(tmp_path, original, "yes")
    changed_server = http()
    changed_server[field] = 2
    changed = {"remote": changed_server}
    write_mcp(tmp_path / "workspace" / ".mycode" / "mcp.json", changed)
    _, updated = load_layers(tmp_path, project=changed)
    prompts = []

    decide(
        tmp_path,
        updated,
        input_func=lambda prompt: prompts.append(prompt) or "n",
    )

    assert len(prompts) == 1


def test_http_url_change_invalidates_project_trust(tmp_path) -> None:
    _, original = load_layers(
        tmp_path,
        project={"local": stdio(), "remote": http("https://one.test/mcp")},
    )
    decide(tmp_path, original, "yes")
    changed = {"local": stdio(), "remote": http("https://two.test/mcp")}
    write_mcp(tmp_path / "workspace" / ".mycode" / "mcp.json", changed)
    _, updated = load_layers(tmp_path, project=changed)

    prompts = []
    config = decide(
        tmp_path,
        updated,
        input_func=lambda prompt: prompts.append(prompt) or "n",
    )

    assert config.mcp_servers == {}
    assert len(prompts) == 1


@pytest.mark.parametrize(
    ("first", "changed"),
    [
        ({"one": http()}, {"one": http(), "two": http("https://two.test/mcp")}),
        ({"one": http(), "two": http("https://two.test/mcp")}, {"one": http()}),
        (
            {"one": http(headers={"Authorization": "Bearer ${TOKEN}"})},
            {"one": http(headers={"X-Api-Key": "${TOKEN}"})},
        ),
        (
            {"one": http(headers={"Authorization": "Bearer ${TOKEN}"})},
            {"one": http(headers={"Authorization": "Token ${TOKEN}"})},
        ),
        ({"one": stdio()}, {"one": http()}),
    ],
)
def test_project_mcp_collection_header_or_transport_change_invalidates_trust(
    tmp_path, first, changed
) -> None:
    _, original = load_layers(
        tmp_path,
        project=first,
        environ={"TOKEN": "secret"},
    )
    decide(tmp_path, original, "yes")
    write_mcp(tmp_path / "workspace" / ".mycode" / "mcp.json", changed)
    _, updated = load_layers(
        tmp_path,
        project=changed,
        environ={"TOKEN": "secret"},
    )
    prompts = []

    decide(
        tmp_path,
        updated,
        input_func=lambda prompt: prompts.append(prompt) or "n",
    )

    assert len(prompts) == 1


def test_resolved_http_url_change_invalidates_project_trust(tmp_path) -> None:
    project = {"remote": http("${MCP_URL}")}
    _, original = load_layers(
        tmp_path,
        project=project,
        environ={"MCP_URL": "https://one.test/mcp"},
    )
    decide(tmp_path, original, "yes")
    _, updated = load_layers(
        tmp_path,
        project=project,
        environ={"MCP_URL": "https://two.test/mcp"},
    )
    prompts = []

    decide(
        tmp_path,
        updated,
        input_func=lambda prompt: prompts.append(prompt) or "n",
    )

    assert len(prompts) == 1


def test_resolved_http_header_token_rotation_does_not_invalidate_trust(tmp_path) -> None:
    project = {
        "remote": http(headers={"Authorization": "Bearer ${SECRET_TOKEN}"})
    }
    _, original = load_layers(
        tmp_path,
        project=project,
        user_env="SECRET_TOKEN=first-secret\n",
    )
    decide(tmp_path, original, "yes")
    _, updated = load_layers(
        tmp_path,
        project=project,
        user_env="SECRET_TOKEN=second-secret\n",
    )

    config = decide(
        tmp_path,
        updated,
        input_func=lambda prompt: pytest.fail("header token rotation must not prompt"),
    )

    assert set(config.mcp_servers) == {"remote"}


def test_legacy_stdio_only_digest_cannot_authorize_added_project_http(tmp_path) -> None:
    workspace, stdio_only = load_layers(tmp_path, project={"local": stdio()})
    legacy = _legacy_stdio_fingerprint(stdio_only)
    assert legacy == project_mcp_fingerprint(stdio_only)
    (tmp_path / "mcp-trust.json").write_text(
        json.dumps(
            {
                "version": 1,
                "projects": {ProjectIdentity.from_workspace(workspace).key: legacy},
            }
        ),
        encoding="utf-8",
    )
    changed = {"local": stdio(), "remote": http()}
    write_mcp(workspace / ".mycode" / "mcp.json", changed)
    _, loaded = load_layers(tmp_path, project=changed)
    prompts = []

    config = decide(
        tmp_path,
        loaded,
        input_func=lambda prompt: prompts.append(prompt) or "n",
    )

    assert config.mcp_servers == {}
    assert len(prompts) == 1


def test_rejected_project_http_override_falls_back_to_user_alias(tmp_path) -> None:
    _, loaded = load_layers(
        tmp_path,
        user={"shared": stdio("trusted-user")},
        project={"shared": http()},
    )

    config = decide(tmp_path, loaded, "n")

    assert config.mcp_servers["shared"].command == "trusted-user"


def test_corrupt_trust_store_fails_closed_and_warns(tmp_path) -> None:
    _, loaded = load_layers(tmp_path, project={"local": stdio()})
    (tmp_path / "mcp-trust.json").write_text("{broken", encoding="utf-8")
    prompts = []
    outputs = []

    config = decide(
        tmp_path,
        loaded,
        input_func=lambda prompt: prompts.append(prompt) or "n",
        outputs=outputs,
    )

    assert config.mcp_servers == {}
    assert len(prompts) == 1
    assert any("信任状态文件无效" in line for line in outputs)


def test_invalid_trust_store_schema_fails_closed(tmp_path) -> None:
    _, loaded = load_layers(tmp_path, project={"local": stdio()})
    (tmp_path / "mcp-trust.json").write_text(
        json.dumps({"version": 2, "projects": {}}), encoding="utf-8"
    )
    prompts = []

    config = decide(
        tmp_path,
        loaded,
        input_func=lambda prompt: prompts.append(prompt) or "n",
    )

    assert config.mcp_servers == {}
    assert len(prompts) == 1


def test_write_failure_allows_current_run_and_warns(tmp_path, monkeypatch) -> None:
    _, loaded = load_layers(tmp_path, project={"local": stdio()})
    outputs = []

    def fail_write(path, store) -> None:
        raise PermissionError

    monkeypatch.setattr("mycode.mcp.trust._write_trust_store", fail_write)
    config = decide(tmp_path, loaded, "yes", outputs=outputs)

    assert set(config.mcp_servers) == {"local"}
    assert any("信任状态未能保存" in line for line in outputs)


def test_prompt_escapes_control_characters_and_never_reveals_resolved_secrets(
    tmp_path,
) -> None:
    _, loaded = load_layers(
        tmp_path,
        project={
            "local": stdio(
                "${COMMAND}",
                args=["line\nbreak", "${ARG_SECRET}"],
                env={"TOKEN": "${ENV_SECRET}"},
            )
        },
        environ={
            "COMMAND": "resolved-command-secret",
            "ARG_SECRET": "resolved-arg-secret",
            "ENV_SECRET": "resolved-env-secret",
        },
    )
    outputs = []

    decide(tmp_path, loaded, outputs=outputs)
    rendered = "\n".join(outputs)

    assert "'${COMMAND}'" in rendered
    assert "line\\nbreak" in rendered
    assert "ARG_SECRET" in rendered
    assert "TOKEN" in rendered
    assert "resolved-command-secret" not in rendered
    assert "resolved-arg-secret" not in rendered
    assert "resolved-env-secret" not in rendered


def test_http_prompt_shows_safe_destination_and_never_reveals_secrets(tmp_path) -> None:
    _, loaded = load_layers(
        tmp_path,
        project={
            "remote": http(
                "${MCP_URL}",
                headers={"Authorization": "Bearer ${SECRET_TOKEN}"},
            )
        },
        user_env=(
            "SECRET_TOKEN=super-secret-token\n"
            "MCP_URL=https://user:password@api.example.test/mcp"
            "?token=super-secret-token#private\n"
        ),
    )
    outputs = []
    (tmp_path / "mcp-trust.json").write_text("{broken", encoding="utf-8")

    decide(tmp_path, loaded, "n", outputs=outputs)
    rendered = "\n".join(outputs)

    assert "server> 'remote'" in rendered
    assert "transport> streamable_http" in rendered
    assert "url template> '${MCP_URL}'" in rendered
    assert "destination> 'https://api.example.test'" in rendered
    assert "header keys> ['Authorization']" in rendered
    assert "信任状态文件无效" in rendered
    assert "super-secret-token" not in rendered
    assert "user:password" not in rendered
    assert "?token=" not in rendered
    assert "#private" not in rendered


def test_http_prompt_sanitizes_userinfo_query_and_fragment_from_literal_url(
    tmp_path,
) -> None:
    _, loaded = load_layers(
        tmp_path,
        project={
            "remote": http(
                "https://user:password@api.example.test:8443/mcp"
                "?token=query-secret#fragment-secret"
            )
        },
    )
    outputs = []

    decide(tmp_path, loaded, "n", outputs=outputs)
    rendered = "\n".join(outputs)

    assert "url template> 'https://api.example.test:8443/mcp'" in rendered
    assert "destination> 'https://api.example.test:8443'" in rendered
    assert "user:password" not in rendered
    assert "query-secret" not in rendered
    assert "fragment-secret" not in rendered


def test_trust_store_does_not_save_resolved_http_url_or_header_secret(tmp_path) -> None:
    workspace, loaded = load_layers(
        tmp_path,
        project={
            "remote": http(
                "${MCP_URL}",
                headers={"Authorization": "Bearer ${SECRET_TOKEN}"},
            )
        },
        user_env=(
            "SECRET_TOKEN=super-secret-token\n"
            "MCP_URL=https://api.example.test/mcp?token=url-secret\n"
        ),
    )

    decide(tmp_path, loaded, "yes")
    stored = (tmp_path / "mcp-trust.json").read_text(encoding="utf-8")

    assert ProjectIdentity.from_workspace(workspace).key in stored
    assert project_mcp_fingerprint(loaded) in stored
    assert "super-secret-token" not in stored
    assert "url-secret" not in stored
    assert "api.example.test" not in stored


def test_trust_store_contains_only_project_key_and_digest_not_secrets(tmp_path) -> None:
    workspace, loaded = load_layers(
        tmp_path,
        project={
            "local": stdio(
                "${COMMAND}",
                args=["${ARG_SECRET}"],
                env={"TOKEN": "literal-env-secret"},
            )
        },
        environ={"COMMAND": "secret-command", "ARG_SECRET": "secret-arg"},
    )

    decide(tmp_path, loaded, "yes")
    stored = (tmp_path / "mcp-trust.json").read_text(encoding="utf-8")

    assert ProjectIdentity.from_workspace(workspace).key in stored
    assert project_mcp_fingerprint(loaded) in stored
    assert "secret-command" not in stored
    assert "secret-arg" not in stored
    assert "literal-env-secret" not in stored
