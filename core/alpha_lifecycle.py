"""
Formal alpha lifecycle state machine.

Every alpha passes through defined states from generation to submission.
Transitions are validated — disallowed transitions raise ValueError.
"""

from enum import StrEnum
from typing import Dict, Set


class AlphaState(StrEnum):
    """Formal states in the alpha lifecycle."""

    GENERATED = "generated"       # expression exists, not yet simulated
    SIMULATING = "simulating"     # submitted to WQ API, awaiting completion
    SIMULATED = "simulated"       # simulation complete, results stored locally
    CHECKED = "checked"           # correlation/quality checks passed
    SUBMITTED = "submitted"       # submitted to WQ for production scoring
    FAILED = "failed"             # simulation or check failed
    RETRYING = "retrying"         # in retry queue


VALID_TRANSITIONS: Dict[AlphaState, Set[AlphaState]] = {
    AlphaState.GENERATED:  {AlphaState.SIMULATING, AlphaState.FAILED},
    AlphaState.SIMULATING: {AlphaState.SIMULATED, AlphaState.FAILED, AlphaState.RETRYING},
    AlphaState.RETRYING:   {AlphaState.SIMULATING, AlphaState.FAILED},
    AlphaState.SIMULATED:  {AlphaState.CHECKED, AlphaState.SUBMITTED,
                            AlphaState.FAILED, AlphaState.RETRYING},
    AlphaState.CHECKED:    {AlphaState.SUBMITTED, AlphaState.FAILED,
                            AlphaState.RETRYING},
    AlphaState.SUBMITTED:  set(),   # terminal
    AlphaState.FAILED:     {AlphaState.GENERATED, AlphaState.RETRYING},
}


def validate_transition(from_state: AlphaState, to_state: AlphaState):
    """Raise ValueError if the transition is not allowed."""
    allowed = VALID_TRANSITIONS.get(from_state, set())
    if to_state not in allowed:
        raise ValueError(
            f"Invalid state transition: {from_state.value} -> "
            f"{to_state.value}. Allowed: "
            f"{[s.value for s in sorted(allowed, key=lambda s: s.value)]}"
        )


def can_transition(from_state: AlphaState, to_state: AlphaState) -> bool:
    """Check whether the transition is allowed."""
    return to_state in VALID_TRANSITIONS.get(from_state, set())
