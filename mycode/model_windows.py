"""Small context-window policy, using only this connection's API declarations."""
from dataclasses import dataclass
import hashlib
import json

from mycode.config import DEFAULT_LLM_CONTEXT_WINDOW_TOKENS


def _connection_digest(connection):
    payload = [connection.base_url, connection.api_key, sorted(connection.headers.items())]
    return hashlib.sha256(json.dumps(payload).encode()).hexdigest()


def connection_scope(reference, connection):
    if reference != "env" and not reference.startswith("@"):
        return reference
    # env/bootstrap/injected references can denote a different connection on the next
    # startup. Store a digest, never their credentials, as the settings key.
    return reference + ":" + _connection_digest(connection)


def window_binding(scope, model, connection):
    # Also invalidate Session-local values when a saved ID is edited.
    return json.dumps([scope, model, _connection_digest(connection)], separators=(",", ":"))


def api_window(metadata):
    limits = [metadata.get(name) for name in ("context_window", "context_window_tokens")]
    limits = [value for value in limits if type(value) is int and value > 0]
    return min(limits) if limits else None


@dataclass(frozen=True)
class ModelWindow:
    requested: int | None
    server_limit: int | None = None

    @property
    def tokens(self):
        if self.server_limit is not None:
            return min(self.requested, self.server_limit) if self.requested is not None else self.server_limit
        return self.requested if self.requested is not None else DEFAULT_LLM_CONTEXT_WINDOW_TOKENS

    @property
    def source(self):
        if self.server_limit is not None:
            return "user/API cap" if self.requested is not None else "API"
        return "user" if self.requested is not None else "default"

    def describe(self):
        verification = f"API limit={self.server_limit}" if self.server_limit is not None else "unverified"
        return f"window={self.tokens} source={self.source} ({verification})"
