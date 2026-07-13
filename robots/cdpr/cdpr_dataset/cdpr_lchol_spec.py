from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from rl_vla_bootstrapping.lchol.embodiment_spec import AchievedOption, HindsightBCRecord

from .rl_instruction_tasks import canonical_object_name


_PHASE_SUCCESS_BONUS = 10.0
_WRONG_CONTACT_BLOCK_THRESHOLD = 0.20

_DIRECT_MOVE_OPTIONS = frozenset(
    {"move_left", "move_right", "move_top", "move_bottom", "move_up", "move_down"}
)
_PUSH_OPTIONS = frozenset({"push_left", "push_right", "push_forward", "push_backward"})
_PLACEMENT_SOURCE_OPTIONS = frozenset(
    {
        "put_into_plate",
        "put_into_bowl",
        "move_left_of_object",
        "move_right_of_object",
        "move_between_objects",
    }
)


class CDPRLCHOLSpec:
    option_names: tuple[str, ...] = (
        "move_to_object",
        "grab_object",
        "pick_up",
        "push_left",
        "push_right",
        "push_forward",
        "push_backward",
        "put_into_plate",
        "put_into_bowl",
        "move_left_of_object",
        "move_right_of_object",
        "move_in_front_of_object",
        "move_behind_object",
        "put_in_front_of_object",
        "put_behind_object",
        "move_between_objects",
        "move_left",
        "move_right",
        "move_top",
        "move_bottom",
        "move_up",
        "move_down",
        "put_near_reference",
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

        if option in {"push_left", "push_right", "push_forward", "push_backward"}:
            if option in {"push_left", "push_right"}:
                sign = -1.0 if option == "push_left" else 1.0
                motion = _float(info.get("target_motion_x"), 0.0)
            else:
                sign = 1.0 if option == "push_forward" else -1.0
                motion = _float(info.get("target_motion_y"), 0.0)
            target_motion_xy = _float(info.get("target_motion_xy"), abs(motion))
            progress = _clip01(sign * motion / max(_float(info.get("push_success_displacement"), 0.08), 1e-6))
            wrong_direction = _clip01((-sign * motion) / max(abs(motion), 0.08, 1e-6))
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

        if option in {"put_into_plate", "put_into_bowl"}:
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

        if option in {
            "move_left_of_object",
            "move_right_of_object",
            "move_in_front_of_object",
            "move_behind_object",
            "put_in_front_of_object",
            "put_behind_object",
            "move_between_objects",
        }:
            relation = _relation_score(info, default_scale=0.20)
            motion_ok = 1.0 if _boolish(info.get("relation_motion_ok", True)) else 0.0
            grasp_ok = 1.0 if _boolish(info.get("relation_grasp_ok", True)) else 0.0
            wrong_direction = _wrong_relation_direction(info, option)
            progress = (
                + 0.15 * approach
                + 0.25 * float(caught_target)
                + 0.35 * relation
                + 0.15 * motion_ok
                + 0.10 * grasp_ok
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
        achieved: dict[tuple[str, str], AchievedOption] = {}
        for timestep, raw_info in enumerate(trajectory):
            info = dict(raw_info)
            target = str(info.get("target_object_catalog") or info.get("target_object_name") or "")
            reference = str(info.get("reference_object_catalog") or "")
            second_reference = str(info.get("second_reference_object_catalog") or "")
            source_instruction = str(info.get("source_instruction") or info.get("language_instruction") or "")

            if _move_to_predicate(info):
                key = ("move_to_object", target)
                if key not in achieved:
                    achieved[key] = AchievedOption(
                        option_name="move_to_object",
                        first_timestep=timestep,
                        instruction=self._instruction_text("move_to_object", target, reference, second_reference),
                        target_object=target,
                        predicate_value=max(0.0, 1.0 - _float(info.get("distance_ee_to_object_xy"), 1.0)),
                        metadata={"source_instruction": source_instruction},
                    )

            for direct_option, displacement in _direct_move_achievements(info):
                key = (direct_option, "")
                if key in achieved:
                    continue
                achieved[key] = AchievedOption(
                    option_name=direct_option,
                    first_timestep=timestep,
                    instruction=self._instruction_text(direct_option, target, reference, second_reference),
                    predicate_value=float(displacement),
                    metadata={"source_instruction": source_instruction},
                )

            if ("grab_object", target) not in achieved and _grab_predicate(info):
                achieved[("grab_object", target)] = AchievedOption(
                    option_name="grab_object",
                    first_timestep=timestep,
                    instruction=self._instruction_text("grab_object", target, reference, second_reference),
                    target_object=target,
                    predicate_value=1.0,
                    metadata={"source_instruction": source_instruction},
                )

            if ("pick_up", target) not in achieved and _pick_predicate(info):
                achieved[("pick_up", target)] = AchievedOption(
                    option_name="pick_up",
                    first_timestep=timestep,
                    instruction=self._instruction_text("pick_up", target, reference, second_reference),
                    target_object=target,
                    predicate_value=1.0,
                    metadata={"source_instruction": source_instruction},
                )

            for push, pushed_object, predicate_value in _push_achievements(info):
                key = (push, pushed_object)
                if key in achieved:
                    continue
                achieved[key] = AchievedOption(
                    option_name=push,
                    first_timestep=timestep,
                    instruction=self._instruction_text(push, pushed_object, reference, second_reference),
                    target_object=pushed_object,
                    predicate_value=float(predicate_value),
                    metadata={"source_instruction": source_instruction},
                )

            if _near_reference_predicate(info):
                key = ("put_near_reference", target)
                if key not in achieved:
                    achieved[key] = AchievedOption(
                        option_name="put_near_reference",
                        first_timestep=timestep,
                        instruction=self._instruction_text(
                            "put_near_reference",
                            target,
                            reference,
                            second_reference,
                        ),
                        target_object=target,
                        reference_object=reference,
                        predicate_value=max(0.0, 1.0 - _target_reference_xy_distance(info)),
                        metadata={"source_instruction": source_instruction},
                    )

            relation = _relation_achievement(info)
            if relation and (relation, target) not in achieved:
                achieved[(relation, target)] = AchievedOption(
                    option_name=relation,
                    first_timestep=timestep,
                    instruction=self._instruction_text(relation, target, reference, second_reference),
                    target_object=target,
                    reference_object=reference,
                    second_reference_object=second_reference,
                    predicate_value=1.0,
                    metadata={"source_instruction": source_instruction},
                )

        return sorted(
            achieved.values(),
            key=lambda item: (item.first_timestep, item.option_name, item.target_object),
        )

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
            if option.option_name not in _allowed_relabel_options(original_option):
                continue
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
        if option == "push_forward":
            return f"push {target_text} forward"
        if option == "push_backward":
            return f"push {target_text} backward"
        if option == "put_into_plate":
            return f"put {target_text} on {reference_text or 'plate'}"
        if option == "put_into_bowl":
            return f"put {target_text} into {reference_text or 'bowl'}"
        if option == "move_left_of_object":
            return f"put {target_text} to the left of {reference_text}"
        if option == "move_right_of_object":
            return f"put {target_text} to the right of {reference_text}"
        if option == "move_in_front_of_object":
            return f"move {target_text} in front of {reference_text}"
        if option == "move_behind_object":
            return f"move {target_text} behind {reference_text}"
        if option == "put_in_front_of_object":
            return f"put {target_text} in front of {reference_text}"
        if option == "put_behind_object":
            return f"put {target_text} behind {reference_text}"
        if option == "move_between_objects":
            return f"put {target_text} between {reference_text} and {second_text}"
        if option == "move_left":
            return "move left"
        if option == "move_right":
            return "move right"
        if option == "move_top":
            return "move forward"
        if option == "move_bottom":
            return "move backward"
        if option == "move_up":
            return "move up"
        if option == "move_down":
            return "move down"
        if option == "put_near_reference":
            return f"put {target_text} near {reference_text}"
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
    if option not in {
        "move_left_of_object",
        "move_right_of_object",
        "move_in_front_of_object",
        "move_behind_object",
        "put_in_front_of_object",
        "put_behind_object",
    }:
        return 0.0
    signed = _float(info.get("signed_relation_offset"), 0.0)
    return 1.0 if signed < -1e-6 else 0.0


def _wrong_direction_motion(info: Mapping[str, Any], option: str) -> float:
    if option in {"push_left", "push_right", "push_forward", "push_backward"}:
        if option in {"push_left", "push_right"}:
            sign = -1.0 if option == "push_left" else 1.0
            motion = _float(info.get("target_motion_x"), 0.0)
        else:
            sign = 1.0 if option == "push_forward" else -1.0
            motion = _float(info.get("target_motion_y"), 0.0)
        return _clip01((-sign * motion) / max(abs(motion), 0.08, 1e-6))
    return _wrong_relation_direction(info, option)


def _grab_predicate(info: Mapping[str, Any]) -> bool:
    if _wrong_object_contact(info) >= _WRONG_CONTACT_BLOCK_THRESHOLD:
        return False
    if _boolish(info.get("caught_object_is_target")) or _boolish(info.get("target_grasped")):
        return True
    if _boolish(info.get("grab_require_caught", True)):
        return False
    threshold = max(_float(info.get("grab_xy_tolerance"), 0.025), 1e-6)
    return bool(_boolish(info.get("gripper_closed")) and _float(info.get("distance_ee_to_object_xy"), 1.0) <= threshold)


def _pick_predicate(info: Mapping[str, Any]) -> bool:
    if _wrong_object_contact(info) >= _WRONG_CONTACT_BLOCK_THRESHOLD:
        return False
    if not (_boolish(info.get("target_grasped")) or _boolish(info.get("caught_object_is_target"))):
        return False
    lift = _float(info.get("pick_target_lift"), 0.0)
    threshold = max(_float(info.get("pick_lift_success_height"), 0.05), 1e-6)
    return lift >= threshold


def _allowed_relabel_options(source_option: str) -> frozenset[str]:
    source = str(source_option or "")
    if source == "move_to_object":
        return frozenset({*_DIRECT_MOVE_OPTIONS, *_PUSH_OPTIONS})
    if source in _PUSH_OPTIONS:
        return frozenset({"move_to_object", *_DIRECT_MOVE_OPTIONS, *_PUSH_OPTIONS})
    if source in _PLACEMENT_SOURCE_OPTIONS:
        return frozenset(
            {
                "move_to_object",
                "grab_object",
                "pick_up",
                "put_near_reference",
                *_DIRECT_MOVE_OPTIONS,
                *_PUSH_OPTIONS,
            }
        )
    # Preserve the broader OpenVLA LC-HOL behavior for its legacy option set.
    return frozenset(CDPRLCHOLSpec.option_names)


def _move_to_predicate(info: Mapping[str, Any]) -> bool:
    distance = _float(
        info.get("distance_ee_to_object_xy", info.get("distance_to_goal_xy")),
        1.0,
    )
    threshold = max(
        _float(
            info.get(
                "move_to_object_validation_distance_threshold",
                info.get("move_to_object_xy_tolerance", 0.03),
            ),
            0.03,
        ),
        1e-6,
    )
    z_ok = _boolish(
        info.get(
            "move_to_object_validation_z_excursion_ok",
            info.get("move_to_object_z_excursion_ok", True),
        ),
        True,
    )
    return bool(distance <= threshold and z_ok)


def _direct_move_achievements(info: Mapping[str, Any]) -> list[tuple[str, float]]:
    current = np.asarray(info.get("ee_position", ()), dtype=np.float32).reshape(-1)
    start = np.asarray(info.get("ee_start", ()), dtype=np.float32).reshape(-1)
    if current.size < 3 or start.size < 3:
        return []
    delta = current[:3] - start[:3]
    threshold = max(_float(info.get("lchol_direction_displacement_threshold"), 0.05), 1e-6)
    mapping = (
        ("move_left", 0, -1.0),
        ("move_right", 0, 1.0),
        ("move_bottom", 1, -1.0),
        ("move_top", 1, 1.0),
        ("move_down", 2, -1.0),
        ("move_up", 2, 1.0),
    )
    return [
        (name, float(sign * delta[axis]))
        for name, axis, sign in mapping
        if float(sign * delta[axis]) >= threshold
    ]


def _target_reference_xy_distance(info: Mapping[str, Any]) -> float:
    target = np.asarray(info.get("target_object_position_actual", ()), dtype=np.float32).reshape(-1)
    reference = np.asarray(info.get("reference_object_position", ()), dtype=np.float32).reshape(-1)
    if target.size < 2 or reference.size < 2:
        return float("inf")
    return float(np.linalg.norm(target[:2] - reference[:2]))


def _near_reference_predicate(info: Mapping[str, Any]) -> bool:
    source = str(info.get("instruction_type") or "")
    if source not in _PLACEMENT_SOURCE_OPTIONS:
        return False
    threshold = max(_float(info.get("lchol_near_reference_tolerance"), 0.16), 1e-6)
    return bool(_target_reference_xy_distance(info) <= threshold)


def _push_achievement(info: Mapping[str, Any]) -> str:
    if _wrong_object_contact(info) >= _WRONG_CONTACT_BLOCK_THRESHOLD:
        return ""
    motion_x = _float(info.get("target_motion_x"), 0.0)
    motion_y = _float(info.get("target_motion_y"), 0.0)
    threshold = max(_float(info.get("push_success_displacement"), 0.08), 0.02)
    if motion_x >= threshold:
        return "push_right"
    if motion_x <= -threshold:
        return "push_left"
    if motion_y >= threshold:
        return "push_forward"
    if motion_y <= -threshold:
        return "push_backward"
    return ""


def _push_achievements(info: Mapping[str, Any]) -> list[tuple[str, str, float]]:
    out: list[tuple[str, str, float]] = []
    target = str(info.get("target_object_catalog") or info.get("target_object_name") or "")
    target_push = _push_achievement(info)
    if target_push:
        motion_key = "target_motion_y" if target_push in {"push_forward", "push_backward"} else "target_motion_x"
        out.append((target_push, target, abs(_float(info.get(motion_key), 0.0))))

    names = [str(item) for item in (info.get("scene_objects") or ())]
    current = np.asarray(info.get("all_object_positions", ()), dtype=np.float32)
    initial = np.asarray(info.get("initial_all_object_positions", ()), dtype=np.float32)
    if current.ndim != 2 or initial.ndim != 2:
        return out
    count = min(len(names), current.shape[0], initial.shape[0])
    threshold = max(_float(info.get("push_success_displacement"), 0.08), 0.02)
    for index in range(count):
        name = names[index]
        if not name or name == target:
            continue
        delta = current[index, :2] - initial[index, :2]
        candidates = (
            ("push_right", float(delta[0])),
            ("push_left", float(-delta[0])),
            ("push_forward", float(delta[1])),
            ("push_backward", float(-delta[1])),
        )
        for option, signed_motion in candidates:
            if signed_motion >= threshold:
                out.append((option, name, signed_motion))
                break
    return out


def _relation_achievement(info: Mapping[str, Any]) -> str:
    instruction = str(info.get("instruction_type") or "")
    if instruction in {"put_into_plate", "put_into_bowl"} and (
        _success(info) >= 0.5 or _relation_score(info, default_scale=0.16) >= 0.75
    ):
        return instruction
    if instruction == "move_between_objects" and _success(info) >= 0.5:
        return "move_between_objects"
    if instruction in {
        "move_left_of_object",
        "move_right_of_object",
        "move_in_front_of_object",
        "move_behind_object",
        "put_in_front_of_object",
        "put_behind_object",
    } and _success(info) >= 0.5:
        return instruction
    signed = _float(info.get("signed_relation_offset"), 0.0)
    axis = int(round(_float(info.get("relation_axis"), 0.0)))
    sign = _float(info.get("relation_axis_sign"), 1.0)
    offset_key = "relation_front_behind_offset" if axis == 1 else "relation_left_right_offset"
    offset = max(_float(info.get(offset_key), _float(info.get("relation_left_right_offset"), 0.08)), 1e-6)
    if signed >= offset and _boolish(info.get("relation_motion_ok", True)):
        if axis == 1:
            return "move_behind_object" if sign > 0.0 else "move_in_front_of_object"
        return "move_right_of_object" if sign > 0.0 else "move_left_of_object"
    if signed <= -offset and _boolish(info.get("relation_motion_ok", True)):
        if axis == 1:
            return "move_in_front_of_object" if sign > 0.0 else "move_behind_object"
        return "move_left_of_object" if sign > 0.0 else "move_right_of_object"
    return ""
