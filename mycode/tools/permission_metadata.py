from mycode.permissions import PermissionDecision


def with_permission_metadata(
    decision: PermissionDecision,
    metadata: dict[str, object],
) -> PermissionDecision:
    merged_metadata = {**decision.metadata, **metadata}

    if decision.status == "allow":
        return PermissionDecision.allow(
            message=decision.message,
            metadata=merged_metadata,
        )

    if decision.status == "ask":
        return PermissionDecision.ask(
            reason=decision.reason,
            message=decision.message,
            metadata=merged_metadata,
        )

    return PermissionDecision.deny(
        reason=decision.reason,
        message=decision.message,
        metadata=merged_metadata,
    )
