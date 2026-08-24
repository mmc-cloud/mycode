from typing import Literal


ReasoningState = Literal[
    "absent",
    "present_empty",
    "present_nonempty",
]
_REASONING_STATES = {
    "absent",
    "present_empty",
    "present_nonempty",
}


def normalize_reasoning(
    state: ReasoningState,
    content: str | None,
) -> tuple[ReasoningState, str | None]:
    if state not in _REASONING_STATES:
        raise ValueError(f"Unsupported reasoning state: {state}")

    if content == "":
        content = None
        if state == "absent":
            state = "present_empty"

    if content is not None:
        if state == "absent":
            state = "present_nonempty"
        elif state != "present_nonempty":
            raise ValueError(
                "Non-empty reasoning content requires present_nonempty state."
            )
    elif state == "present_nonempty":
        raise ValueError(
            "present_nonempty reasoning state requires non-empty content."
        )

    return state, content
