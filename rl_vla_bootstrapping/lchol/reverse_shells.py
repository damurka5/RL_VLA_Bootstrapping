from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol


@dataclass(frozen=True)
class ReverseShellReset:
    instruction_id: str
    shell_id: int
    metadata: Mapping[str, Any] = field(default_factory=dict)


class ReverseShellSpec(Protocol):
    instruction_id: str
    instruction_template: str
    shell_count: int

    def sample_scene(self, rng: Any) -> Mapping[str, Any]:
        """Return scene randomization constraints for this instruction family."""

    def sample_reset(self, shell_id: int, scene: Any, rng: Any, **kwargs: Any) -> ReverseShellReset:
        """Apply or describe a reverse-curriculum reset for one shell."""

    def success(self, state: Any, instruction_binding: Mapping[str, Any]) -> bool:
        """Sparse terminal predicate used for reward and validation."""


def clamp_shell_id(shell_id: int, shell_count: int) -> int:
    if int(shell_count) <= 0:
        raise ValueError("shell_count must be positive.")
    return max(0, min(int(shell_id), int(shell_count) - 1))
