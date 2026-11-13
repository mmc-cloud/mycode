"""Connection commands and atomic changes of an existing application's models."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json

from pydantic import ValidationError

from mycode.application.runtime import context_budget_from_config
from mycode.context.budget import TokenEstimator
from mycode.model_activity import ModelActivity
from mycode.model_selection import ENV_CONNECTION, injected_reference, ModelSelection, RoleSelection
from mycode.model_config import ModelConfiguration
from mycode.model_windows import ModelWindow, api_window, connection_scope, window_binding
from mycode.persistence.connection_store import ConnectionStore, ConnectionStoreError, SavedConnection
from mycode.providers.factory import ModelClientFactory


def _close_client(client):
    try:
        client._client.close()
    except Exception:
        pass  # Cleanup cannot roll back an already committed switch.


def _same_model(first, second):
    # Compare effective wire settings, excluding input-only metadata and
    # unrelated role names in LLMConfig. Never log this credential-bearing tuple.
    return all(getattr(first.config, name) == getattr(second.config, name) for name in (
        "api_key", "base_url", "headers", "stream_include_usage", "max_output_tokens",
    )) and all(getattr(first, name) == getattr(second, name) for name in (
        "model", "thinking_enabled", "reasoning_effort",
    ))


def close_model_clients(runner):
    _close_client(runner.llm_client)
    if runner.compactor is not None:
        _close_client(runner.compactor.llm_client)


class SessionModels:
    def __init__(self, runner, active_session, store=None, *, explicit=False):
        self.runner, self.session = runner, active_session
        inputs = runner.model_configuration
        self.connections = dict(inputs.connections)
        self.store = ConnectionStore() if store is None else store
        self.activity = ModelActivity(validate=self._validate_active_window)
        self._closed = False
        self.lease = None
        self.catalogs = {}
        self.catalog_connections = {}
        selection = inputs.selection
        if explicit:
            updates = {}
            for role in ("main", "subagent", "compact"):
                setting = getattr(selection, role)
                ref, model = selection.identity(role)
                connection = inputs.connections[ref]
                ref = injected_reference(ref)
                values = {"connection_ref": ref} if setting.connection_ref is not None else {}
                if role in inputs.windows:
                    window = inputs.windows[role]
                    scope = connection_scope(ref, connection)
                    # A bound None preserves an explicit automatic window,
                    # rather than following Main's manual preference.
                    values.update(context_window_tokens=window.requested,
                                  window_binding=window_binding(scope, model, connection))
                    if window.server_limit is not None:
                        self.catalogs.setdefault(ref, {})[model] = {"context_window": window.server_limit}
                        self.catalog_connections[ref] = connection
                updates[role] = setting.model_copy(update=values)
            selection = selection.model_copy(update=updates)
            self.connections = {injected_reference(ref): value for ref, value in self.connections.items()}
        runner.compactor.model_activity = self.activity
        delegate = runner.tool_registry.require("delegate_task")
        delegate.runtime = replace(delegate.runtime, model_activity=self.activity)
        runner.model_configuration = self._window_configuration(selection, None)
        if explicit:
            if any(self.windows[role].tokens != inputs.resolve(role).context_window_tokens
                   for role in ("main", "subagent", "compact")):
                self.switch(self.selection)
            else:
                self.session.writer.save_model_selection(self.selection)
            return
        restored = self.session.writer.load_model_selection()
        if restored is not None:
            self.switch(restored, persist=False)
            return
        doc = self.store.read()
        if doc.default_models is not None:
            self.switch(doc.default_models)
        elif doc.default_ref is not None:
            main = self.selection.main.model_copy(update={"connection_ref": doc.default_ref,
                "thinking": "auto", "effort": "auto", "max_output_tokens": "auto",
                "context_window_tokens": None, "window_binding": None})
            self.switch(self.selection.model_copy(update={"main": main}))
        else:
            candidate = self._window_configuration(self.selection, doc)
            if any(candidate.windows[role].tokens != self.windows[role].tokens
                   for role in ("main", "subagent", "compact")):
                self.switch(candidate.selection)
            else:
                runner.model_configuration = candidate

    @property
    def selection(self):
        return self.runner.model_configuration.selection

    @property
    def windows(self):
        return self.runner.model_configuration.windows

    def close(self):
        self._closed = True
        close_model_clients(self.runner)
        if self.lease is not None:
            self.lease.close()
            self.lease = None

    def _validate_active_window(self, role):
        setting = getattr(self.selection, role)
        ref, model = setting.connection_ref or self.selection.main.connection_ref, setting.model or self.selection.main.model
        limit = api_window(self.catalogs.get(ref, {}).get(model, {}))
        if limit is not None and self.windows[role].tokens > limit:
            raise RuntimeError("API context limit changed; choose compatible /model budgets or start a new Session before requesting this role.")

    def switch(self, selection, *, persist=True, window_edit=None):
        if self._closed:
            raise RuntimeError("Model settings belong to a closed Session.")
        with self.activity.change():
            self._switch(selection, persist=persist, window_edit=window_edit)

    def _window_configuration(self, selection, doc, window_edit=None):
        available = {} if doc is None else {key: value.connection() for key, value in doc.connections.items()}
        available.update(self.connections)
        identities = {}
        for role in ("main", "subagent", "compact"):
            setting = getattr(selection, role)
            ref, model = setting.connection_ref or selection.main.connection_ref, setting.model or selection.main.model
            if ref not in available:
                raise ConnectionStoreError("Saved connection is unavailable; choose an existing connection.")
            scope = connection_scope(ref, available[ref])
            identities[role] = (ref, model, scope, window_binding(scope, model, available[ref]))
            if ref in self.catalog_connections and self.catalog_connections[ref] != available[ref]:
                self.catalogs.pop(ref, None)
                self.catalog_connections.pop(ref, None)
        if window_edit is not None:
            role, value = window_edit
            ref, model, scope, binding = identities[role]
            limit = api_window(self.catalogs.get(ref, {}).get(model, {}))
            if value is not None and limit is not None and value > limit:
                raise ConnectionStoreError("Configured context window exceeds the known server limit.")
            values = doc.model_windows.setdefault(scope, {})
            if value is None:
                values.pop(model, None)
                if not values:
                    doc.model_windows.pop(scope, None)
            else:
                values[model] = value
        updates, windows, manuals = {}, {}, {}
        for role, (ref, model, scope, binding) in identities.items():
            setting = getattr(selection, role)
            bound = setting.window_binding == binding
            local = setting.context_window_tokens if setting.window_binding in {None, binding} else None
            if window_edit is not None and binding == identities[window_edit[0]][3]:
                local = None
                bound = False
            saved = doc.model_windows.get(scope, {}).get(model) if doc is not None else None
            manual = saved if saved is not None else local
            if manual is None and not bound and role != "main" and binding == identities["main"][3]:
                manual = manuals["main"]
            manuals[role] = manual
            limit = api_window(self.catalogs.get(ref, {}).get(model, {}))
            windows[role] = ModelWindow(manual, limit)
            # Shared preferences have one owner: connections.json. Session data
            # retains only genuine env/injected local intent.
            local = None if saved is not None else local
            updates[role] = setting.model_copy(update={"context_window_tokens": local,
                "window_binding": binding if saved is None and (local is not None or bound) else None})
        selection = selection.model_copy(update=updates)
        return ModelConfiguration(selection=selection, connections=available, windows=windows,
            memory_context_tokens=self.runner.model_configuration.memory_context_tokens,
            stream_include_usage=self.runner.model_configuration.stream_include_usage)

    def _switch(self, selection, *, persist=True, window_edit=None):
        with self.store.transaction():
            doc = self.store.read()
            original_windows = deepcopy(doc.model_windows)
            selection = ModelSelection.model_validate(selection)
            models = self._window_configuration(selection, doc, window_edit)
            selection = models.selection
            factory = ModelClientFactory(models, self.runner.llm_client.session_id)
            created = []
            lease = None
            try:
                for role in ("main", "compact", "subagent"):
                    self._validate_known_model(selection, factory, role)
                for role in ("main", "compact"):
                    created.append(factory.create(role))
                main_budget = context_budget_from_config(factory.config_for("main"))
                current = self.runner.inspect_context()
                if current.estimate.estimated_input_tokens > main_budget.max_input_tokens:
                    raise ConnectionStoreError("History exceeds the target input budget; Compact or start a new Session first.")
                delegate = self.runner.tool_registry.require("delegate_task")
                sub_runtime = replace(delegate.runtime,
                    llm_client_factory=lambda: factory.create("subagent"),
                    context_budget=context_budget_from_config(factory.config_for("subagent")),
                    model_activity=self.activity)
                lease = self.store._pin(selection.references() - self.connections.keys())
                changed_windows = doc.model_windows != original_windows
                if changed_windows:
                    self.store._write(doc)
                try:
                    if persist:
                        self.session.writer.save_model_selection(selection)
                except BaseException:
                    if changed_windows:
                        doc.model_windows = original_windows
                        self.store._write(doc)
                    raise
                old_main, old_summary = self.runner.llm_client, self.runner.compactor.llm_client
                old_lease = self.lease
                main_changed = not _same_model(old_main, created[0])
                summary_changed = not _same_model(old_summary, created[1])
                self.runner.llm_client = created[0]
                self.runner.context_budget = main_budget
                delegate.runtime = sub_runtime
                self.runner.compactor.llm_client = created[1]
                self.runner.compactor.summary_budget = context_budget_from_config(factory.config_for("compact"))
                if summary_changed:
                    self.runner.compactor.summary_token_estimator = TokenEstimator()
                if main_changed:
                    self.runner.token_estimator = TokenEstimator()
                    self.runner.last_token_usage = None
                    self.runner.last_model_context = None
                self.runner.model_configuration = models
                self.lease = lease
                lease = None
            except BaseException:
                for client in created:
                    _close_client(client)
                if lease is not None:
                    lease.close()
                raise
            _close_client(old_main)
            _close_client(old_summary)
            if old_lease is not None:
                old_lease.close()

    def _validate_known_model(self, selection, factory, role):
        setting = getattr(selection, role)
        ref = setting.connection_ref or selection.main.connection_ref
        catalog = self.catalogs.get(ref)
        known = (catalog or {}).get(factory.config_for(role).model)
        if known is None:
            return
        config = factory.config_for(role)
        declaration = known.get("effort")
        effort = declaration.get("supported_levels") if isinstance(declaration, dict) else None
        if effort and config.reasoning_effort is not None and config.reasoning_effort not in effort:
            raise ConnectionStoreError("Selected model does not support this reasoning effort.")
        output = known.get("max_output_tokens")
        if type(output) is int and output > 0 and config.max_output_tokens is not None and config.max_output_tokens > output:
            raise ConnectionStoreError("Configured output limit exceeds the known server limit.")

    def status(self):
        lines = []
        for role in ("main", "subagent", "compact"):
            selected = getattr(self.selection, role)
            ref = selected.connection_ref or self.selection.main.connection_ref
            model = selected.model or self.selection.main.model
            intent = "fixed connection/model" if selected.connection_ref else ("fixed model" if selected.model else "follow Main")
            lines.append(f"{role}: connection={ref} model={model} ({intent}); thinking={selected.thinking or 'follow/default'} effort={selected.effort or 'auto'}; {self.windows[role].describe()}")
        return tuple(lines)

    def command(self, name, args, *, values=None):
        if self._closed:
            raise RuntimeError("Model settings belong to a closed Session.")
        try:
            if name == "model":
                return self._model(args)
            if not args or args == ("list",):
                doc = self.store.read()
                return ("env: environment connection (credentials hidden)", *(
                    f"{key}{' [default for new sessions]' if key == doc.default_ref else ''}: {value.base_url} (credentials hidden)"
                    for key, value in sorted(doc.connections.items())
                ))
            action = args[0]
            if action == "use" and len(args) >= 3:
                return self._model(("main", args[2], f"connection={args[1]}", *args[3:]))
            if action == "models" and len(args) == 2:
                return self._discover(args[1])
            if action in {"add", "edit"} and len(args) == 2 and values is not None:
                with self.activity.change():
                    identifier = args[1]
                    old = self.store.read().connections.get(identifier)
                    data = {} if old is None else old.model_dump()
                    data.update(id=identifier)
                    for key in ("base_url", "api_key", "headers"):
                        value = values.get(key)
                        if value not in (None, ""):
                            data[key] = json.loads(value) if key == "headers" and isinstance(value, str) else value
                    saved = SavedConnection.model_validate(data)
                    self.store.save(saved, edit=action == "edit")
                    self.catalogs.pop(identifier, None)
                return ("Connection saved; credentials hidden. Use /connect use ID MODEL to activate it.",)
            if action == "delete" and len(args) == 2:
                with self.activity.change():
                    self.store.delete(args[1])
                return ("Connection deleted.",)
            if action == "default" and len(args) == 2:
                with self.activity.change():
                    self.store.set_default(None if args[1] == "env" else args[1])
                return ("Default connection saved for new Sessions; active Sessions were not changed.",)
            raise ConnectionStoreError("Usage: /connect [list | add ID | edit ID | delete ID | default ID | models ID | use ID MODEL [options]]")
        except (ValidationError, json.JSONDecodeError, TypeError):
            raise ConnectionStoreError("Invalid connection/model parameters; no configuration change was applied.") from None

    def _model(self, args):
        if not args:
            return self.status()
        if args in {("default",), ("default", "clear")}:
            with self.activity.change():
                self.store.set_model_defaults(None if len(args) == 2 else self.selection)
            return ("Default role settings cleared." if len(args) == 2 else "Current role settings saved for new Sessions; existing Sessions are unchanged.",)
        if len(args) < 2 or args[0] not in {"main", "subagent", "compact"}:
            raise ConnectionStoreError("Usage: /model ROLE MODEL|follow [connection=ID|follow thinking=auto|on|off effort=low|high|max window=N reserve=N margin=N output=N]")
        role, model = args[:2]
        options = {}
        names = {"connection", "thinking", "effort", "window", "reserve", "margin", "output"}
        for argument in args[2:]:
            key, sep, value = argument.partition("=")
            if not sep or key in options or key not in names:
                raise ConnectionStoreError("Options must be known, unique key=value pairs.")
            if (key != "connection" and value == "inherit") or (key == "connection" and value == "follow"):
                value = None
            elif key in {"window", "reserve", "margin", "output"} and value != "auto":
                try:
                    value = int(value)
                except ValueError:
                    raise ConnectionStoreError("Budget options must be integers or inherit.") from None
            options[key] = value
        return self.select_model(role, None if model == "follow" else model, **options)

    def select_model(self, role, model, **options):
        """Apply typed options shared by interactive and machine adapters."""
        if role not in {"main", "subagent", "compact"}:
            raise ConnectionStoreError("Unknown model role.")
        if model is None and role == "main":
            raise ConnectionStoreError("Only SubAgent/Compact can follow Main.")
        names = {"connection": "connection_ref", "thinking": "thinking", "effort": "effort",
                 "window": "context_window_tokens", "reserve": "reserved_output_tokens",
                 "margin": "context_safety_margin_tokens", "output": "max_output_tokens"}
        if options.keys() - names.keys():
            raise ConnectionStoreError("Unknown model option.")
        if model is None and not options:
            chosen = RoleSelection()
        else:
            old = getattr(self.selection, role)
            values = old.model_dump()
            values["model"] = model
            if model is None:
                values["connection_ref"] = None
            if "connection" in options:
                values["connection_ref"] = options["connection"]
            if model != old.model or values["connection_ref"] != old.connection_ref:
                values.update(thinking="off" if role == "compact" else "auto", effort="auto",
                              max_output_tokens="auto")
            if "thinking" in options and options["thinking"] is not None:
                values["effort"] = "auto"
            values.update({names[key]: value for key, value in options.items()})
            chosen = RoleSelection.model_validate(values)
        candidate = self.selection.model_copy(update={role: chosen})
        window_edit = (role, chosen.context_window_tokens) if "window" in options else None
        self.switch(candidate, window_edit=window_edit)
        return (*self.status(), "Model selection saved. Window sources are shown above; manual/default windows without an API limit are unverified. Use window=inherit to clear a saved window.")

    def _discover(self, identifier):
        rows = self.discover_models(identifier)
        if rows is None:
            return ("Model listing unavailable; enter a model ID manually (unverified). Previously returned limits remain in effect for this connection.",)
        if not rows:
            return ("No models listed; enter a model ID manually (unverified). Previously returned limits remain in effect for this connection.",)
        return tuple(f"{identifier}: {row['id']}; " + (
            f"API context window={row['context_window']}" if row["context_window"] is not None
            else "context window unavailable (unverified)") for row in rows)

    def discover_models(self, identifier):
        """Query once, apply known limits, and expose only public model fields."""
        if self._closed:
            raise RuntimeError("Model settings belong to a closed Session.")
        # A successful query may change active budgets, so use the same idle
        # boundary as /model rather than racing an in-flight model request.
        with self.activity.change():
            doc = self.store.read()
            connection = self.connections.get(identifier)
            if connection is None:
                saved = doc.connections.get(identifier)
                if saved is None:
                    raise ConnectionStoreError("Connection is unavailable; configure it before use.")
                connection = saved.connection()
            from mycode.providers.openai_compatible import OpenAICompatibleLLMClient
            config = replace(self.runner.llm_client.config, api_key=connection.api_key, base_url=connection.base_url, headers=connection.headers)
            client = OpenAICompatibleLLMClient(config, session_id=self.runner.llm_client.session_id)
            try:
                result = client._client.models.list()
                rows = {item.id: item.model_dump() for item in result.data}
            except Exception:
                return None
            finally:
                _close_client(client)
            if not rows:
                return []
            self.catalogs[identifier] = rows
            self.catalog_connections[identifier] = connection
            active_refs = {getattr(self.selection, role).connection_ref or self.selection.main.connection_ref
                           for role in ("main", "subagent", "compact")}
            if identifier in active_refs:
                self._switch(self.selection)
            return [{"id": key, "context_window": api_window(rows[key])} for key in sorted(rows)]
