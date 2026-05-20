from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from rl_vla_bootstrapping.lchol.embodiment_spec import AchievedOption, HindsightBCRecord

from .rl_instruction_tasks import canonical_object_name


_PHASE_SUCCESS_BONUS = 10.0
_WRONG_CONTACT_BLOCK_THRESHOLD = 0.20


class CDPRLCHOLSpec:
    option_names: tuple[str, ...] = (
        "move_to_object",
        "grab_object",
        "pick_up",
        "push_left",
        "push_right",
        "put_into_plate",
        "move_left_of_object",
        "move_right_of_object",
        "move_between_objects",
    )

    def phase_score(self, trajectory: Sequence[Mapping[str, Any]], option_name: str | None = None) -> float:
        if not trajectory:
            return 0.0
        info = dict(trajectory[-1])
        option = str(option_name or info.get("instruction_type") or "").strip()
        sparse_success = _success(info)
        approach = _approach_score(info)
        caught_target = _boolish(info.get("caught_object_is_target")) or _boolish(info.get("target_grasped"))
        gripper_closed = _boolish(info.get("gripper_closed"))
        wrong_contact = _wrong_object_contact(info)
        premature = _premature_stop(info, sparse_success=sparse_success)
        saturation = _action_saturation(info)

        if option in {"grab_object", "pick_up"}:
            lift = _clip01(_float(info.get("pick_target_lift_normalized"), 0.0))
            grasp = 1.0 if caught_target else _clip01(_float(info.get("pick_contact_score"), 0.0))
            progress = (
                + 0.30 * approach
                + 0.30 * float(gripper_closed) * max(approach, grasp)
                + 0.30 * float(caught_target)
                + 0.10 * lift
                - 0.20 * wrong_contact
                - 0.20 * premature
                - 0.20 * saturation
            )
            return _phase_total_score(sparse_success=sparse_success, progress=progress)

        if option in {"push_left", "push_right"}:
            sign = -1.0 if option == "push_left" else 1.0
            motion_x = _float(info.get("target_motion_x"), 0.0)
            target_motion_xy = _float(info.get("target_motion_xy"), abs(motion_x))
            progress = _clip01(sign * motion_x / max(_float(info.get("push_success_displacement"), 0.08), 1e-6))
            wrong_direction = _clip01((-sign * motion_x) / max(abs(motion_x), 0.08, 1e-6))
            contact = max(_clip01(target_motion_xy / 0.08), 1.0 if caught_target else 0.0)
            shaped = (
                + 0.20 * approach
                + 0.25 * contact
                + 0.45 * progress
                - 0.25 * wrong_direction
                - 0.20 * premature
                - 0.20 * saturation
            )
            return _phase_total_score(sparse_success=sparse_success, progress=shaped)

        if option == "put_into_plate":
            relation = _relation_score(info, default_scale=0.16)
            release = relation * (1.0 - float(gripper_closed) if "gripper_closed" in info else 1.0)
            off_plate = 1.0 - relation if _boolish(info.get("env_done")) and not sparse_success else 0.0
            progress = (
                + 0.15 * approach
                + 0.25 * float(caught_target)
                + 0.35 * relation
                + 0.25 * release
                - 0.20 * off_plate
                - 0.20 * premature
                - 0.20 * saturation
            )
            return _phase_total_score(sparse_success=sparse_success, progress=progress)

        if option in {"move_left_of_object", "move_right_of_object", "move_between_objects"}:
            relation = _relation_score(info, default_scale=0.20)
            motion_ok = 1.0 if _boolish(info.get("relation_motion_ok", True)) else 0.0
            wrong_direction = _wrong_relation_direction(info, option)
            progress = (
                + 0.15 * approach
                + 0.25 * float(caught_target)
                + 0.35 * relation
                + 0.25 * motion_ok
                - 0.20 * wrong_direction
                - 0.20 * premature
                - 0.20 * saturation
            )
            return _phase_total_score(sparse_success=sparse_success, progress=progress)

        return _phase_total_score(
            sparse_success=sparse_success,
            progress=0.50 * approach - 0.20 * premature - 0.20 * saturation,
        )

    def achieved_options(self, trajectory: Sequence[Mapping[str, Any]]) -> list[AchievedOption]:
        achieved: dict[str, AchievedOption] = {}
        for timestep, raw_info in enumerate(trajectory):
            info = dict(raw_info)
            target = str(info.get("target_object_catalog") or info.get("target_object_name") or "")
            reference = str(info.get("reference_object_catalog") or "")
            second_reference = str(info.get("second_reference_object_catalog") or "")
            source_instruction = str(info.get("source_instruction") or info.get("language_instruction") or "")

            if "grab_object" not in achieved and _grab_predicate(info):
                achieved["grab_object"] = AchievedOption(
                    option_name="grab_object",
                    first_timestep=timestep,
                    instruction=self._instruction_text("grab_object", target, reference, second_reference),
                    target_object=target,
                    predicate_value=1.0,
                    metadata={"source_instruction": source_instruction},
                )

            if "pick_up" not in achieved and _pick_predicate(info):
                achieved["pick_up"] = AchievedOption(
                    option_name="pick_up",
                    first_timestep=timestep,
                    instruction=self._instruction_text("pick_up", target, reference, second_reference),
                    target_object=target,
                    predicate_value=1.0,
                    metadata={"source_instruction": source_instruction},
                )

            push = _push_achievement(info)
            if push and push not in achieved:
                achieved[push] = AchievedOption(
                    option_name=push,
                    first_timestep=timestep,
                    instruction=self._instruction_text(push, target, reference, second_reference),
                    target_object=target,
                    predicate_value=abs(_float(info.get("target_motion_x"), 0.0)),
                    metadata={"source_instruction": source_instruction},
                )

            relation = _relation_achievement(info)
            if relation and relation not in achieved:
                achieved[relation] = AchievedOption(
                    option_name=relation,
                    first_timestep=timestep,
                    instruction=self._instruction_text(relation, target, reference, second_reference),
                    target_object=target,
                    reference_object=reference,
                    second_reference_object=second_reference,
                    predicate_value=1.0,
                    metadata={"source_instruction": source_instruction},
                )

        return sorted(achieved.values(), key=lambda item: (item.first_timestep, item.option_name))

    def relabel_instruction(self, achieved_option: AchievedOption) -> str:
        if achieved_option.instruction:
            return achieved_option.instruction
        return self._instruction_text(
            achieved_option.option_name,
            achieved_option.target_object,
            achieved_option.reference_object,
            achieved_option.second_reference_object,
        )

    def synthetic_completion_action(self, achieved_option: AchievedOption, state: Mapping[str, Any]) -> Any:
        del achieved_option, state
        return None

    def build_hindsight_records(
        self,
        trajectory: Sequence[Mapping[str, Any]],
        *,
        prefix_max_steps: int,
    ) -> list[HindsightBCRecord]:
        achieved = self.achieved_options(trajectory)
        records: list[HindsightBCRecord] = []
        for option in achieved:
            end = int(option.first_timestep) + 1
            start = max(0, end - max(1, int(prefix_max_steps)))
            prefix = [dict(step) for step in trajectory[start:end]]
            if not prefix:
                continue
            final = prefix[-1]
            action = final.get("action")
            if action is None:
                continue
            instruction = self.relabel_instruction(option)
            source_instruction = str(
                final.get("source_instruction")
                or final.get("language_instruction")
                or option.metadata.get("source_instruction", "")
            )
            if _normalise_instruction_text(source_instruction) == _normalise_instruction_text(instruction):
                continue
            original_option = str(final.get("instruction_type") or final.get("option_name") or "")
            prefix_start = int(start)
            prefix_end = int(end - 1)
            records.append(
                HindsightBCRecord(
                    option_name=option.option_name,
                    instruction=instruction,
                    action=action,
                    source_instruction=source_instruction,
                    first_timestep=option.first_timestep,
                    image_primary=final.get("image_primary"),
                    image_wrist=final.get("image_wrist"),
                    prefix_actions=tuple(step.get("action") for step in prefix if step.get("action") is not None),
                    metadata={
                        "source_rollout_id": final.get("source_rollout_id", final.get("rollout_id", "")),
                        "source_policy_version": final.get("source_policy_version", final.get("policy_version", "")),
                        "original_instruction": source_instruction,
                        "relabeled_instruction": instruction,
                        "original_option": original_option,
                        "relabeled_option": option.option_name,
                        "target_object": option.target_object,
                        "reference_object": option.reference_object,
                        "second_reference_object": option.second_reference_object,
                        "first_achievement_timestep": int(option.first_timestep),
                        "prefix_start_timestep": prefix_start,
                        "prefix_end_timestep": prefix_end,
                        "predicate_name": option.option_name,
                        "predicate_value": float(option.predicate_value),
                        "predicate_margin": float(option.predicate_value),
                        "wrong_object_contact": float(_wrong_object_contact(final)),
                        "wrong_direction_motion": float(_wrong_direction_motion(final, option.option_name)),
                        "sparse_success_original": float(_success(final)),
                        "sparse_success_relabel": 1.0,
                        "video_or_frame_path": final.get("video_or_frame_path", final.get("frame_path", "")),
                    },
                )
            )
        return records

    def _instruction_text(self, option: str, target: str, reference: str, second_reference: str) -> str:
        target_text = canonical_object_name(target)
        reference_text = canonical_object_name(reference)
        second_text = canonical_object_name(second_reference)
        if option == "move_to_object":
            return f"move to {target_text}"
        if option == "grab_object":
            return f"grab {target_text}"
        if option == "pick_up":
            return f"pick up {target_text}"
        if option == "push_left":
            return f"push {target_text} left"
        if option == "push_right":
            return f"push {target_text} right"
        if option == "put_into_plate":
            return f"put {target_text} into {reference_text or 'plate'}"
        if option == "move_left_of_object":
            return f"move {target_text} to the left of {reference_text}"
        if option == "move_right_of_object":
            return f"move {target_text} to the right of {reference_text}"
        if option == "move_between_objects":
            return f"move {target_text} between {reference_text} and {second_text}"
        return option.replace("_", " ")


def _float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if np.isfinite(out) else float(default)


def _boolish(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    try:
        return bool(float(value) >= 0.5)
    except (TypeError, ValueError):
        return bool(value)


def _clip01(value: float) -> float:
    return float(np.clip(float(value), 0.0, 1.0))


def _phase_total_score(*, sparse_success: float, progress: float) -> float:
    return float(_PHASE_SUCCESS_BONUS * _clip01(sparse_success) + _clip01(progress))


def _normalise_instruction_text(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


def _success(info: Mapping[str, Any]) -> float:
    if "sparse_success" in info:
        return 1.0 if _boolish(info.get("sparse_success")) else 0.0
    return 1.0 if _boolish(info.get("success")) else 0.0


def _approach_score(info: Mapping[str, Any]) -> float:
    if "distance_ee_to_object_xy" not in info and "distance_to_goal_xy" not in info:
        return 0.0
    dist = _float(info.get("distance_ee_to_object_xy", info.get("distance_to_goal_xy")), 1.0)
    scale = max(_float(info.get("lchol_approach_scale"), 0.25), 1e-6)
    return _clip01(1.0 - dist / scale)


def _relation_score(info: Mapping[str, Any], *, default_scale: float) -> float:
    relation_error = _float(info.get("relation_error"), default_scale)
    if relation_error < 0.0:
        return 0.0
    scale = max(_float(info.get("lchol_relation_error_scale"), default_scale), 1e-6)
    return _clip01(1.0 - relation_error / scale)


def _wrong_object_contact(info: Mapping[str, Any]) -> float:
    if _boolish(info.get("caught_object_is_target")):
        return 0.0
    return _clip01(_float(info.get("caught_object_score"), 0.0))


def _action_saturation(info: Mapping[str, Any]) -> float:
    if "action_saturation_rate" in info:
        return _clip01(_float(info.get("action_saturation_rate"), 0.0))
    if "action_saturation_penalty_raw" in info:
        return _clip01(_float(info.get("action_saturation_penalty_raw"), 0.0))
    return 0.0


def _premature_stop(info: Mapping[str, Any], *, sparse_success: float) -> float:
    if sparse_success >= 0.5:
        return 0.0
    if _boolish(info.get("forced_unstable_reset")) or _boolish(info.get("unstable_transition")):
        return 1.0
    if _boolish(info.get("env_done")) and not _boolish(info.get("success")):
        return 1.0
    return 0.0


def _wrong_relation_direction(info: Mapping[str, Any], option: str) -> float:
    if option not in {"move_left_of_object", "move_right_of_object"}:
        return 0.0
    signed = _float(info.get("signed_relation_offset"), 0.0)
    if option == "move_left_of_object":
        return 1.0 if signed > 1e-6 else 0.0
    return 1.0 if signed < -1e-6 else 0.0


def _wrong_direction_motion(info: Mapping[str, Any], option: str) -> float:
    if option in {"push_left", "push_right"}:
        sign = -1.0 if option == "push_left" else 1.0
        motion_x = _float(info.get("target_motion_x"), 0.0)
        return _clip01((-sign * motion_x) / max(abs(motion_x), 0.08, 1e-6))
    return _wrong_relation_direction(info, option)


def _grab_predicate(info: Mapping[str, Any]) -> bool:
    if _wrong_object_contact(info) >= _WRONG_CONTACT_BLOCK_THRESHOLD:
        return False
    if _boolish(info.get("caught_object_is_target")) or _boolish(info.get("target_grasped")):
        return True
    return bool(_boolish(info.get("gripper_closed")) and _float(info.get("distance_ee_to_object_xy"), 1.0) <= 0.045)


def _pick_predicate(info: Mapping[str, Any]) -> bool:
    if _wrong_object_contact(info) >= _WRONG_CONTACT_BLOCK_THRESHOLD:
        return False
    if not (_boolish(info.get("target_grasped")) or _boolish(info.get("caught_object_is_target"))):
        return False
    lift = _float(info.get("pick_target_lift"), 0.0)
    threshold = max(_float(info.get("pick_lift_success_height"), 0.05), 1e-6)
    return lift >= threshold


def _push_achievement(info: Mapping[str, Any]) -> str:
    if _wrong_object_contact(info) >= _WRONG_CONTACT_BLOCK_THRESHOLD:
        return ""
    motion_x = _float(info.get("target_motion_x"), 0.0)
    threshold = max(_float(info.get("push_success_displacement"), 0.08), 0.02)
    if motion_x >= threshold:
        return "push_right"
    if motion_x <= -threshold:
        return "push_left"
    return ""


def _relation_achievement(info: Mapping[str, Any]) -> str:
    instruction = str(info.get("instruction_type") or "")
    if instruction == "put_into_plate" and (_success(info) >= 0.5 or _relation_score(info, default_scale=0.16) >= 0.75):
        return "put_into_plate"
    if instruction == "move_between_objects" and _success(info) >= 0.5:
        return "move_between_objects"
    if instruction in {"move_left_of_object", "move_right_of_object"} and _success(info) >= 0.5:
        return instruction
    signed = _float(info.get("signed_relation_offset"), 0.0)
    offset = max(_float(info.get("relation_left_right_offset"), 0.08), 1e-6)
    if signed >= offset and _boolish(info.get("relation_motion_ok", True)):
        return "move_right_of_object"
    if signed <= -offset and _boolish(info.get("relation_motion_ok", True)):
        return "move_left_of_object"
    return ""
