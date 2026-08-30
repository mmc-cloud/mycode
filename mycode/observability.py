"""Content-free structured observability for Agent and Harbor runs."""

from collections.abc import Callable, Mapping
from typing import Literal


OBSERVABILITY_SCHEMA_VERSION = 1
ObservationSink = Callable[[dict[str, object]], None]
CompletionDecisionValue = Literal["accepted", "correction", "rejected"]


def emit_observation(
    sink: ObservationSink | None,
    event_type: str,
    fields: Mapping[str, object],
) -> None:
    """Emit one versioned record without exposing model or tool bodies."""
    if sink is None:
        return
    try:
        sink(
            {
                "schema_version": OBSERVABILITY_SCHEMA_VERSION,
                "event_type": event_type,
                **fields,
            }
        )
    except Exception:
        # Observability must never change Agent control flow or task outcome.
        return
