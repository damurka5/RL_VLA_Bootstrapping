"""Shared full-task outcome observer, independent of training milestones."""

from dataclasses import dataclass
from typing import Any

from rl_vla_bootstrapping.policy.cdpr_staged_demonstrations import contact_ended_without_release


@dataclass(frozen=True)
class FullTaskOutcome:
    native: Any
    grasped: Any
    lifted: Any
    released: Any
    carry_slip: Any
    wrong_place: Any

    @classmethod
    def zeros(cls, torch: Any, worlds: int, device: Any) -> "FullTaskOutcome":
        empty = torch.zeros(worlds, dtype=torch.bool, device=device)
        return cls(*(empty.clone() for _ in range(6)))

    @property
    def strict(self) -> Any:
        return (self.native & self.grasped & self.lifted & self.released
                & ~self.carry_slip & ~self.wrong_place)

    def advance(self, *, active: Any, native_success: Any, physical_grasp: Any,
                held_lift: Any, released: Any, release_in_progress: Any,
                wrong_place: Any) -> "FullTaskOutcome":
        native = self.native | (active & native_success)
        grasped = self.grasped | (active & physical_grasp)
        lifted = self.lifted | (active & held_lift)
        ever_released = self.released | (active & released & grasped)
        # Use the current release signal and exempt completed native placements,
        # exactly as the standalone full-task evaluator has always done.
        slip = self.carry_slip | (
            active & lifted & ~native & contact_ended_without_release(
                physical_grasp=physical_grasp, released=released,
                release_in_progress=release_in_progress,
            )
        )
        return FullTaskOutcome(native, grasped, lifted, ever_released, slip,
                               self.wrong_place | (active & wrong_place))
