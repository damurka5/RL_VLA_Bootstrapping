from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rl_vla_bootstrapping.policy.action_codec import ActionCodec, QuantizationSpec


def _as_str_list(values: Any) -> list[str]:
    if values is None:
        return []
    return [str(item) for item in values]


def _as_float_pair_map(values: dict[str, Any] | None) -> dict[str, tuple[float, float]]:
    out: dict[str, tuple[float, float]] = {}
    for key, raw in (values or {}).items():
        if raw is None or len(raw) != 2:
            raise ValueError(f"Expected `[low, high]` for `{key}`, got {raw!r}")
        out[str(key)] = (float(raw[0]), float(raw[1]))
    return out


@dataclass(frozen=True)
class EntrypointRef:
    attribute: str
    file: str | None = None
    module: str | None = None
    python_paths: list[str] = field(default_factory=list)

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | None) -> "EntrypointRef | None":
        if not data:
            return None
        attribute = data.get("attribute") or data.get("class_name") or data.get("function")
        if not attribute:
            raise ValueError("Entrypoint needs `attribute`, `class_name`, or `function`.")
        return cls(
            attribute=str(attribute),
            file=str(data["file"]) if data.get("file") else None,
            module=str(data["module"]) if data.get("module") else None,
            python_paths=_as_str_list(data.get("python_paths")),
        )


@dataclass(frozen=True)
class ProjectSpec:
    name: str
    output_root: str = "runs"
    python_executable: str = "python3"
    env: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | None) -> "ProjectSpec":
        data = data or {}
        name = str(data.get("name") or "bootstrap_run")
        return cls(
            name=name,
            output_root=str(data.get("output_root") or "runs"),
            python_executable=str(data.get("python_executable") or "python3"),
            env={str(k): str(v) for k, v in (data.get("env") or {}).items()},
        )


@dataclass(frozen=True)
class ReposSpec:
    openvla_oft: str | None = None
    dataset_repo: str | None = None
    embodiment_repo: str | None = None
    benchmark_repos: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | None) -> "ReposSpec":
        data = data or {}
        return cls(
            openvla_oft=str(data["openvla_oft"]) if data.get("openvla_oft") else None,
            dataset_repo=str(data["dataset_repo"]) if data.get("dataset_repo") else None,
            embodiment_repo=str(data["embodiment_repo"]) if data.get("embodiment_repo") else None,
            benchmark_repos={str(k): str(v) for k, v in (data.get("benchmark_repos") or {}).items()},
        )


@dataclass(frozen=True)
class AssetBundleSpec:
    name: str
    source_path: str | None = None
    target_path: str | None = None
    kind: str = "asset"
    required: bool = True
    description: str = ""

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "AssetBundleSpec":
        return cls(
            name=str(data.get("name") or "bundle"),
            source_path=str(data["source_path"]) if data.get("source_path") else None,
            target_path=str(data["target_path"]) if data.get("target_path") else None,
            kind=str(data.get("kind") or "asset"),
            required=bool(data.get("required", True)),
            description=str(data.get("description") or ""),
        )


@dataclass(frozen=True)
class AssetsSpec:
    bundles: tuple[AssetBundleSpec, ...] = field(default_factory=tuple)

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | None) -> "AssetsSpec":
        data = data or {}
        return cls(
            bundles=tuple(AssetBundleSpec.from_mapping(item) for item in (data.get("bundles") or []))
        )


@dataclass(frozen=True)
class RemoteSpec:
    conda_env_file: str | None = None
    env_vars: dict[str, str] = field(default_factory=dict)
    notes: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | None) -> "RemoteSpec":
        data = data or {}
        return cls(
            conda_env_file=str(data["conda_env_file"]) if data.get("conda_env_file") else None,
            env_vars={str(k): str(v) for k, v in (data.get("env_vars") or {}).items()},
            notes=tuple(_as_str_list(data.get("notes"))),
        )


@dataclass(frozen=True)
class ActionAdapterSpec:
    common_action_keys: tuple[str, ...]
    controller_scales: dict[str, float] = field(default_factory=dict)
    controller_limits: dict[str, tuple[float, float]] = field(default_factory=dict)
    policy_low: float = -1.0
    policy_high: float = 1.0
    open_gripper_threshold: float = -0.2
    close_gripper_threshold: float = 0.2

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | None) -> "ActionAdapterSpec":
        data = data or {}
        keys = tuple(_as_str_list(data.get("common_action_keys")))
        if not keys:
            raise ValueError("Embodiment action adapter needs `common_action_keys`.")
        return cls(
            common_action_keys=keys,
            controller_scales={str(k): float(v) for k, v in (data.get("controller_scales") or {}).items()},
            controller_limits=_as_float_pair_map(data.get("controller_limits")),
            policy_low=float(data.get("policy_low", -1.0)),
            policy_high=float(data.get("policy_high", 1.0)),
            open_gripper_threshold=float(data.get("open_gripper_threshold", -0.2)),
            close_gripper_threshold=float(data.get("close_gripper_threshold", 0.2)),
        )


@dataclass(frozen=True)
class ControllerSpec:
    entrypoint: EntrypointRef
    init_kwargs: dict[str, Any] = field(default_factory=dict)
    method_map: dict[str, str] = field(default_factory=dict)
    frame_buffers: dict[str, str] = field(default_factory=dict)
    camera_handles: dict[str, str] = field(default_factory=dict)
    preview_steps: int = 1

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | None) -> "ControllerSpec":
        data = data or {}
        entrypoint = EntrypointRef.from_mapping(data)
        if entrypoint is None:
            raise ValueError("Embodiment controller needs an importable entrypoint.")
        return cls(
            entrypoint=entrypoint,
            init_kwargs=dict(data.get("init_kwargs") or {}),
            method_map={str(k): str(v) for k, v in (data.get("method_map") or {}).items()},
            frame_buffers={str(k): str(v) for k, v in (data.get("frame_buffers") or {}).items()},
            camera_handles={str(k): str(v) for k, v in (data.get("camera_handles") or {}).items()},
            preview_steps=int(data.get("preview_steps", 1)),
        )


@dataclass(frozen=True)
class EmbodimentSpec:
    name: str
    kind: str
    robot_root: str
    xml_path: str
    controller: ControllerSpec
    dof: int
    cameras: dict[str, str] = field(default_factory=dict)
    joint_metadata_path: str | None = None
    collision_config_path: str | None = None
    action_adapter: ActionAdapterSpec = field(default_factory=lambda: ActionAdapterSpec(common_action_keys=("x",)))
    extra_python_paths: list[str] = field(default_factory=list)

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | None) -> "EmbodimentSpec":
        data = data or {}
        return cls(
            name=str(data.get("name") or "unnamed_embodiment"),
            kind=str(data.get("kind") or "mujoco"),
            robot_root=str(data.get("robot_root") or "."),
            xml_path=str(data.get("xml_path") or ""),
            controller=ControllerSpec.from_mapping(data.get("controller")),
            dof=int(data.get("dof") or 0),
            cameras={str(k): str(v) for k, v in (data.get("cameras") or {}).items()},
            joint_metadata_path=str(data["joint_metadata_path"]) if data.get("joint_metadata_path") else None,
            collision_config_path=str(data["collision_config_path"]) if data.get("collision_config_path") else None,
            action_adapter=ActionAdapterSpec.from_mapping(data.get("action_adapter")),
            extra_python_paths=_as_str_list(data.get("extra_python_paths")),
        )


@dataclass(frozen=True)
class TaskSpec:
    instruction_types: tuple[str, ...] = field(default_factory=tuple)
    target_objects: tuple[str, ...] = field(default_factory=tuple)
    reward: EntrypointRef | None = None
    success_predicate: EntrypointRef | None = None
    goal_region: dict[str, Any] = field(default_factory=dict)
    goal_relation: str | None = None
    dense_reward_terms: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | None) -> "TaskSpec":
        data = data or {}
        return cls(
            instruction_types=tuple(_as_str_list(data.get("instruction_types"))),
            target_objects=tuple(_as_str_list(data.get("target_objects"))),
            reward=EntrypointRef.from_mapping(data.get("reward")),
            success_predicate=EntrypointRef.from_mapping(data.get("success_predicate")),
            goal_region=dict(data.get("goal_region") or {}),
            goal_relation=str(data["goal_relation"]) if data.get("goal_relation") else None,
            dense_reward_terms=dict(data.get("dense_reward_terms") or {}),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass(frozen=True)
class SceneBuilderSpec:
    entrypoint: EntrypointRef | None = None
    preview_scene: str | None = None
    preview_objects: tuple[str, ...] = field(default_factory=tuple)
    catalog_path: str | None = None
    object_roots: list[str] = field(default_factory=list)
    desk_textures_dir: str | None = None
    build_kwargs: dict[str, Any] = field(default_factory=dict)
    randomization: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | None) -> "SceneBuilderSpec":
        data = data or {}
        return cls(
            entrypoint=EntrypointRef.from_mapping(data.get("entrypoint")),
            preview_scene=str(data["preview_scene"]) if data.get("preview_scene") else None,
            preview_objects=tuple(_as_str_list(data.get("preview_objects"))),
            catalog_path=str(data["catalog_path"]) if data.get("catalog_path") else None,
            object_roots=_as_str_list(data.get("object_roots")),
            desk_textures_dir=str(data["desk_textures_dir"]) if data.get("desk_textures_dir") else None,
            build_kwargs=dict(data.get("build_kwargs") or {}),
            randomization=dict(data.get("randomization") or {}),
        )


@dataclass(frozen=True)
class ActionCodecSpec:
    chunk_size: int = 8
    normalized_low: float = -1.0
    normalized_high: float = 1.0
    quantization_enabled: bool = True
    quantization_bins: int = 256
    export_filename: str = "action_codec.json"

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | None) -> "ActionCodecSpec":
        data = data or {}
        quant = dict(data.get("quantization") or {})
        return cls(
            chunk_size=int(data.get("chunk_size", 8)),
            normalized_low=float(data.get("normalized_low", -1.0)),
            normalized_high=float(data.get("normalized_high", 1.0)),
            quantization_enabled=bool(quant.get("enabled", data.get("quantization_enabled", True))),
            quantization_bins=int(quant.get("bins", data.get("quantization_bins", 256))),
            export_filename=str(data.get("export_filename") or "action_codec.json"),
        )

    def build_codec(self, action_adapter: ActionAdapterSpec) -> ActionCodec:
        return ActionCodec(
            action_keys=action_adapter.common_action_keys,
            controller_limits=action_adapter.controller_limits,
            normalized_low=self.normalized_low,
            normalized_high=self.normalized_high,
            chunk_size=self.chunk_size,
            quantization=QuantizationSpec(
                enabled=self.quantization_enabled,
                bins=self.quantization_bins,
            ),
        )


@dataclass(frozen=True)
class PolicySpec:
    type: str
    repo_path: str | None = None
    base_checkpoint: str | None = None
    rl_script: str | None = None
    sft_script: str | None = None
    action_head: str | None = None
    num_images_in_input: int = 2
    extra_python_paths: list[str] = field(default_factory=list)
    action_codec: ActionCodecSpec = field(default_factory=ActionCodecSpec)

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | None) -> "PolicySpec":
        data = data or {}
        return cls(
            type=str(data.get("type") or "external_policy"),
            repo_path=str(data["repo_path"]) if data.get("repo_path") else None,
            base_checkpoint=str(data["base_checkpoint"]) if data.get("base_checkpoint") else None,
            rl_script=str(data["rl_script"]) if data.get("rl_script") else None,
            sft_script=str(data["sft_script"]) if data.get("sft_script") else None,
            action_head=str(data["action_head"]) if data.get("action_head") else None,
            num_images_in_input=int(data.get("num_images_in_input", 2)),
            extra_python_paths=_as_str_list(data.get("extra_python_paths")),
            action_codec=ActionCodecSpec.from_mapping(data.get("action_codec")),
        )


@dataclass(frozen=True)
class RLStageSpec:
    enabled: bool = True
    algorithm: str = "ppo"
    script_path: str | None = None
    args: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | None) -> "RLStageSpec":
        data = data or {}
        return cls(
            enabled=bool(data.get("enabled", True)),
            algorithm=str(data.get("algorithm") or "ppo"),
            script_path=str(data["script_path"]) if data.get("script_path") else None,
            args=dict(data.get("args") or {}),
        )


@dataclass(frozen=True)
class SFTStageSpec:
    enabled: bool = False
    script_path: str | None = None
    args: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | None) -> "SFTStageSpec":
        data = data or {}
        return cls(
            enabled=bool(data.get("enabled", False)),
            script_path=str(data["script_path"]) if data.get("script_path") else None,
            args=dict(data.get("args") or {}),
        )


@dataclass(frozen=True)
class TrainingSpec:
    preview_before_rl: bool = True
    rl: RLStageSpec = field(default_factory=RLStageSpec)
    sft: SFTStageSpec = field(default_factory=SFTStageSpec)

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | None) -> "TrainingSpec":
        data = data or {}
        return cls(
            preview_before_rl=bool(data.get("preview_before_rl", True)),
            rl=RLStageSpec.from_mapping(data.get("rl")),
            sft=SFTStageSpec.from_mapping(data.get("sft")),
        )


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    enabled: bool = False
    script_path: str | None = None
    args: dict[str, Any] = field(default_factory=dict)
    extra_python_paths: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "BenchmarkSpec":
        return cls(
            name=str(data.get("name") or "benchmark"),
            enabled=bool(data.get("enabled", False)),
            script_path=str(data["script_path"]) if data.get("script_path") else None,
            args=dict(data.get("args") or {}),
            extra_python_paths=_as_str_list(data.get("extra_python_paths")),
            env={str(k): str(v) for k, v in (data.get("env") or {}).items()},
        )


@dataclass(frozen=True)
class EvaluationSpec:
    benchmarks: tuple[BenchmarkSpec, ...] = field(default_factory=tuple)

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | None) -> "EvaluationSpec":
        data = data or {}
        return cls(
            benchmarks=tuple(BenchmarkSpec.from_mapping(item) for item in (data.get("benchmarks") or []))
        )


@dataclass(frozen=True)
class ProjectConfig:
    config_path: Path
    project: ProjectSpec
    repos: ReposSpec
    remote: RemoteSpec
    assets: AssetsSpec
    embodiment: EmbodimentSpec
    task: TaskSpec
    simulation: SceneBuilderSpec
    policy: PolicySpec
    training: TrainingSpec
    evaluation: EvaluationSpec

    @property
    def root_dir(self) -> Path:
        return self.config_path.parent.resolve()

    def resolve_path(self, raw_path: str | None) -> Path | None:
        if raw_path is None:
            return None
        path = Path(raw_path).expanduser()
        if path.is_absolute():
            return path.resolve()
        return (self.root_dir / path).resolve()

    def build_action_codec(self) -> ActionCodec:
        return self.policy.action_codec.build_codec(self.embodiment.action_adapter)

    def all_python_paths(self) -> list[Path]:
        out: list[Path] = []
        for raw in self.embodiment.extra_python_paths:
            path = self.resolve_path(raw)
            if path is not None:
                out.append(path)
        for raw in self.embodiment.controller.entrypoint.python_paths:
            path = self.resolve_path(raw)
            if path is not None:
                out.append(path)
        for raw in self.policy.extra_python_paths:
            path = self.resolve_path(raw)
            if path is not None:
                out.append(path)
        return out

    def asset_bundle(self, name: str) -> AssetBundleSpec | None:
        for bundle in self.assets.bundles:
            if bundle.name == name:
                return bundle
        return None
