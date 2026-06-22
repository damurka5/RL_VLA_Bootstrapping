from __future__ import annotations

import json
import os
import sys
import tempfile
import types
import unittest
from unittest import mock
from pathlib import Path

import numpy as np

_INSERTED_TORCH_STUB = False

if "torch" not in sys.modules:
    torch_stub = types.ModuleType("torch")

    class _NoGrad:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    torch_stub.Tensor = object
    torch_stub.no_grad = lambda: _NoGrad()
    torch_stub.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch_stub.backends = types.SimpleNamespace(
        cuda=types.SimpleNamespace(matmul=types.SimpleNamespace(allow_tf32=False)),
        cudnn=types.SimpleNamespace(allow_tf32=False),
    )
    torch_stub.set_float32_matmul_precision = lambda *args, **kwargs: None
    sys.modules["torch"] = torch_stub
    _INSERTED_TORCH_STUB = True

if "PIL" not in sys.modules or "PIL.Image" not in sys.modules:
    pil_stub = types.ModuleType("PIL")
    image_stub = types.ModuleType("PIL.Image")

    class _DummyImage:
        def convert(self, mode: str):
            return self

    image_stub.fromarray = lambda arr: _DummyImage()
    pil_stub.Image = image_stub
    sys.modules["PIL"] = pil_stub
    sys.modules["PIL.Image"] = image_stub

import rl_vla_bootstrapping.policy.grpo_finetune_cdpr_fast as grpo_fast
from rl_vla_bootstrapping.policy.grpo_finetune_cdpr_fast import (
    _FastWrapperArgs,
    _LCHOLWrapperArgs,
    _RolloutTensorboardLogger,
    _infer_resume_artifacts,
    _lr_schedule_factor,
    _patch_distributed_timeout,
    _patch_desk_texture_prepare,
    _patch_fresh_scene_cache_prebuild,
    _patch_lchol_runtime,
    _patch_scene_cache_prebuild_progress,
    _patch_scene_wrapper_cache,
    _patch_tensorboard_metric_filter,
    _patch_training_reset_retries,
    _split_wrapper_argv,
    _tensorboard_tag_allowed,
    _transform_external_grpo_source_for_ddp_sync,
    _transform_external_grpo_source_for_lchol,
    _transform_external_grpo_source_for_lr_scheduler,
    _transform_external_grpo_source_for_memory_safety,
)

if _INSERTED_TORCH_STUB:
    sys.modules.pop("torch", None)


class FastGRPOWrapperTests(unittest.TestCase):
    def test_rollout_tensorboard_logger_emits_sparse_manipulation_metrics(self):
        class FakeSummaryWriter:
            scalars: list[tuple[str, float, int]] = []

            def __init__(self, log_dir: str, flush_secs: int = 10):
                self.log_dir = log_dir
                self.flush_secs = flush_secs

            def add_scalar(self, tag: str, value: float, step: int):
                self.scalars.append((tag, float(value), int(step)))

            def flush(self):
                pass

            def close(self):
                pass

        with tempfile.TemporaryDirectory() as tmp:
            FakeSummaryWriter.scalars.clear()
            logger = _RolloutTensorboardLogger(FakeSummaryWriter, every_global_steps=2)
            logger.set_run_dir(Path(tmp))

            for idx in range(2):
                logger.capture_reward(
                    env_reward=1.0,
                    shaped_reward=1.0,
                    closer_bonus=0.0,
                    farther_penalty=0.0,
                    distance_delta_raw=0.0,
                )
                logger.finalize_step(
                    {
                        "success": idx == 1,
                        "sparse_success": idx == 1,
                        "env_done": idx == 1,
                        "env_instance_id": 0,
                        "episode_index": 0,
                        "instruction_type": "pick_up",
                        "curriculum_mode": "reverse_frontier",
                        "curriculum_shell": 2,
                        "distance_ee_to_object_xy": 0.02,
                        "target_motion_xy": 0.05 * idx,
                        "relation_error": 0.08 - 0.02 * idx,
                        "gripper_closed": 1.0,
                        "caught_object_is_target": idx == 1,
                        "grasped": idx == 1,
                        "pick_target_lift": 0.06 if idx == 1 else 0.0,
                        "pick_lift_success_height": 0.05,
                    },
                    {},
                )

            tags = {tag for tag, _, _ in FakeSummaryWriter.scalars}
            self.assertIn("rollout_step/sparse_success_mean", tags)
            self.assertIn("rollout_step/success_rate_mean", tags)
            self.assertIn("rollout_step/episode_success_rate_mean", tags)
            self.assertIn("rollout_step/distance_ee_to_object_xy_mean", tags)
            self.assertIn("rollout_step/relation_error_mean", tags)
            self.assertIn("rollout_step/target_motion_xy_mean", tags)
            self.assertNotIn("rollout_step/window_size", tags)
            self.assertNotIn("rollout_step/reward_component_r_xyz_mean", tags)
            self.assertIn("rollout_episode/instruction_success_rate/pick_up", tags)
            self.assertIn("rollout_episode/instruction_success_rate_mean", tags)
            self.assertIn("rollout_episode/shell_success_rate/pick_up/shell_02", tags)
            self.assertIn("rollout_episode/subgoal_success_rate/move_to_object", tags)
            self.assertIn("rollout_episode/subgoal_success_rate/grab_object", tags)
            self.assertIn("rollout_episode/subgoal_success_rate/pick_up", tags)

    def test_rollout_tensorboard_logger_suppresses_shell_and_subgoal_tags_in_dense_stage(self):
        class FakeSummaryWriter:
            scalars: list[tuple[str, float, int]] = []

            def __init__(self, log_dir: str, flush_secs: int = 10):
                del log_dir, flush_secs

            def add_scalar(self, tag: str, value: float, step: int):
                self.scalars.append((tag, float(value), int(step)))

            def flush(self):
                pass

            def close(self):
                pass

        with tempfile.TemporaryDirectory() as tmp:
            FakeSummaryWriter.scalars.clear()
            logger = _RolloutTensorboardLogger(
                FakeSummaryWriter,
                every_global_steps=1,
                stage_fn=lambda: "dense",
            )
            logger.set_run_dir(Path(tmp))
            logger.capture_reward(
                env_reward=1.0,
                shaped_reward=1.0,
                closer_bonus=0.0,
                farther_penalty=0.0,
                distance_delta_raw=0.0,
            )
            logger.finalize_step(
                {
                    "success": True,
                    "sparse_success": 1.0,
                    "env_done": True,
                    "env_instance_id": 0,
                    "episode_index": 0,
                    "instruction_type": "release_object",
                    "curriculum_mode": "reverse_frontier",
                    "curriculum_shell": 0,
                },
                {},
            )

            tags = {tag for tag, _, _ in FakeSummaryWriter.scalars}
            self.assertIn("rollout_episode/instruction_success_rate/release_object", tags)
            self.assertIn("rollout_episode/instruction_success_rate_mean", tags)
            self.assertNotIn("rollout_episode/shell_success_rate/release_object/shell_00", tags)
            self.assertFalse(any(tag.startswith("rollout_episode/subgoal_success_rate/") for tag in tags))

    def test_rollout_tensorboard_logger_records_partial_subgoal_successes_for_failed_attempt(self):
        class FakeSummaryWriter:
            scalars: list[tuple[str, float, int]] = []

            def __init__(self, log_dir: str, flush_secs: int = 10):
                self.log_dir = log_dir
                self.flush_secs = flush_secs

            def add_scalar(self, tag: str, value: float, step: int):
                self.scalars.append((tag, float(value), int(step)))

            def flush(self):
                pass

            def close(self):
                pass

        with tempfile.TemporaryDirectory() as tmp:
            FakeSummaryWriter.scalars.clear()
            logger = _RolloutTensorboardLogger(FakeSummaryWriter, every_global_steps=2)
            logger.set_run_dir(Path(tmp))

            for idx in range(2):
                logger.capture_reward(
                    env_reward=0.0,
                    shaped_reward=0.0,
                    closer_bonus=0.0,
                    farther_penalty=0.0,
                    distance_delta_raw=0.0,
                )
                logger.finalize_step(
                    {
                        "success": False,
                        "sparse_success": 0.0,
                        "env_done": idx == 1,
                        "env_instance_id": 0,
                        "episode_index": 0,
                        "instruction_type": "put_into_plate",
                        "distance_ee_to_object_xy": 0.01,
                        "gripper_closed": 1.0,
                        "caught_object_is_target": idx == 1,
                        "grasped": idx == 1,
                        "pick_target_lift": 0.08 if idx == 1 else 0.0,
                        "pick_lift_success_height": 0.05,
                    },
                    {},
                )

            scalars = {tag: value for tag, value, _ in FakeSummaryWriter.scalars}
            self.assertEqual(scalars["rollout_episode/instruction_success_rate/put_into_plate"], 0.0)
            self.assertEqual(scalars["rollout_episode/subgoal_success_rate/grab_object"], 1.0)
            self.assertEqual(scalars["rollout_episode/subgoal_success_rate/pick_up"], 1.0)
            self.assertEqual(scalars["rollout_episode/subgoal_success_rate/put_into_plate"], 0.0)

    def test_patch_desk_texture_prepare_uses_single_writer(self):
        calls: list[tuple[str, int]] = []

        def original_prepare(src_dir, run_dir, is_main, rank, max_textures):
            calls.append(("prepare", int(rank)))
            return f"prepared:{rank}"

        def broadcast_object(obj, rank):
            calls.append(("broadcast", int(rank)))
            return "broadcasted"

        module = types.SimpleNamespace(
            _prepare_desk_textures_dir=original_prepare,
            _broadcast_object=broadcast_object,
        )

        _patch_desk_texture_prepare(module)

        self.assertEqual(module._prepare_desk_textures_dir("/tmp/src", Path("/tmp/run"), False, 1, 5), "broadcasted")
        self.assertEqual(module._prepare_desk_textures_dir("/tmp/src", Path("/tmp/run"), True, 0, 5), "prepared:0")
        self.assertEqual(calls, [("broadcast", 1), ("prepare", 0)])

    def test_patch_desk_texture_prepare_wraps_nested_ppo_module(self):
        calls: list[tuple[str, int]] = []

        def original_prepare(src_dir, run_dir, is_main, rank, max_textures):
            del src_dir, run_dir, is_main, max_textures
            calls.append(("prepare", int(rank)))
            return f"prepared:{rank}"

        def broadcast_object(obj, rank):
            del obj
            calls.append(("broadcast", int(rank)))
            return "broadcasted"

        ppo_module = types.SimpleNamespace(
            _prepare_desk_textures_dir=original_prepare,
            _broadcast_object=broadcast_object,
        )
        module = types.SimpleNamespace(ppo=ppo_module)

        _patch_desk_texture_prepare(module)

        self.assertEqual(ppo_module._prepare_desk_textures_dir("/tmp/src", Path("/tmp/run"), False, 1, 5), "broadcasted")
        self.assertEqual(ppo_module._prepare_desk_textures_dir("/tmp/src", Path("/tmp/run"), True, 0, 5), "prepared:0")
        self.assertEqual(calls, [("broadcast", 1), ("prepare", 0)])

    def test_scene_wrapper_cache_filters_variants_by_requested_objects(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            wrong = tmp_path / "wrong.xml"
            right = tmp_path / "right.xml"
            wrong.write_text(
                "<mujoco><worldbody>"
                "<body name='p0_plate'/><body name='p1_ycb_apple'/><body name='p2_ycb_baseball'/>"
                "</worldbody></mujoco>",
                encoding="utf-8",
            )
            right.write_text(
                "<mujoco><worldbody>"
                "<body name='p0_ycb_apple'/><body name='p1_bowl'/><body name='p2_ycb_baseball'/>"
                "</worldbody></mujoco>",
                encoding="utf-8",
            )

            class FakeRL:
                def __init__(self):
                    self._desk_texture_name = ""
                    self._background_color = ""
                    self.np_random = np.random.default_rng(0)

                def _build_wrapper(self, scene, ee_start=None):
                    del scene, ee_start
                    return tmp_path / "rebuilt.xml"

            class FakeVisionEnv:
                def __init__(self):
                    self.env = FakeRL()
                    self._scene_wrapper_cache = {}
                    self._texture_name_by_wrapper = {}

                def _activate_scene_wrapper_cache(self, scene_wrapper_cache, texture_name_by_wrapper):
                    del texture_name_by_wrapper
                    self._scene_wrapper_cache = {str(k): [Path(p).resolve() for p in v] for k, v in scene_wrapper_cache.items()}

                    def cached_builder(rl_self, scene):
                        del rl_self
                        return self._scene_wrapper_cache[str(scene.name)][0]

                    self.env._build_wrapper_original = self.env._build_wrapper
                    self.env._build_wrapper = types.MethodType(cached_builder, self.env)
                    return {}

            module = types.SimpleNamespace(CDPRVisionLanguageEnv=FakeVisionEnv, types=types)
            _patch_scene_wrapper_cache(module)
            env = module.CDPRVisionLanguageEnv()
            env._activate_scene_wrapper_cache({"desk": [wrong, right]}, {})
            scene = types.SimpleNamespace(name="desk", objects=("ycb_apple", "bowl", "ycb_baseball"))

            out = env.env._build_wrapper(scene)

            self.assertEqual(out, right.resolve())
            self.assertEqual(env._scene_wrapper_cache["desk"], [wrong.resolve(), right.resolve()])

    def test_fresh_scene_cache_prebuild_ignores_existing_wrapper_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)

            class FakeRL:
                def __init__(self):
                    self.scenes = [
                        types.SimpleNamespace(name="desk", objects=("ycb_apple", "bowl")),
                        types.SimpleNamespace(name="desk", objects=("ycb_pear", "plate")),
                    ]
                    self.defaults = {"ee_start": [0.0, 0.0, 0.40]}
                    self.desk_texture_files = []
                    self._task_metadata = {}
                    self.use_wrapper_cache = True
                    self.reuse_existing_wrapper_variants = True
                    self.wrapper_cleanup = True
                    self.calls: list[tuple[tuple[str, ...], bool, bool]] = []

                def _build_wrapper(self, scene, ee_start=None):
                    del ee_start
                    objects = tuple(scene.objects)
                    self.calls.append(
                        (objects, bool(self.use_wrapper_cache), bool(self.reuse_existing_wrapper_variants))
                    )
                    path = tmp_path / ("-".join(objects) + ".xml")
                    bodies = "".join(f"<body name='p{idx}_{name}'/>" for idx, name in enumerate(objects))
                    path.write_text(f"<mujoco><worldbody>{bodies}</worldbody></mujoco>", encoding="utf-8")
                    return path

            class FakeVisionEnv:
                _safe_cache_tag = staticmethod(lambda value: str(value))

                def __init__(self):
                    self.env = FakeRL()
                    self._rl_env_module = types.SimpleNamespace()
                    self._scene_wrapper_cache = {}
                    self._texture_name_by_wrapper = {}

                def enable_prebuilt_scene_cache(self, scene_pool_size, texture_pool_size, seed):
                    del scene_pool_size, texture_pool_size, seed
                    raise AssertionError("original stale-cache prebuilder should not run")

                def _activate_scene_wrapper_cache(self, scene_wrapper_cache, texture_name_by_wrapper):
                    self._scene_wrapper_cache = {
                        str(key): [Path(path).resolve() for path in paths]
                        for key, paths in scene_wrapper_cache.items()
                    }
                    self._texture_name_by_wrapper = dict(texture_name_by_wrapper)
                    return {
                        "scenes": len(scene_wrapper_cache),
                        "variants": sum(len(paths) for paths in scene_wrapper_cache.values()),
                        "textures": 0,
                    }

            module = types.SimpleNamespace(CDPRVisionLanguageEnv=FakeVisionEnv)
            _patch_fresh_scene_cache_prebuild(
                module,
                ["--prebuild_scene_cache", "--no-use_wrapper_cache"],
            )
            env = module.CDPRVisionLanguageEnv()

            info = env.enable_prebuilt_scene_cache(scene_pool_size=2, texture_pool_size=0, seed=3)

            self.assertEqual(info["variants"], 2)
            self.assertEqual(len(env._scene_wrapper_cache["desk"]), 2)
            self.assertEqual(
                env.env.calls,
                [
                    (("ycb_apple", "bowl"), False, False),
                    (("ycb_pear", "plate"), False, False),
                ],
            )
            self.assertTrue(env.env.use_wrapper_cache)
            self.assertTrue(env.env.reuse_existing_wrapper_variants)
            self.assertTrue(env.env.wrapper_cleanup)

    def test_training_reset_retries_known_cdpr_reset_failure(self):
        class FakeVisionEnv:
            def __init__(self):
                self.calls = 0

            def reset(self, options=None):
                del options
                self.calls += 1
                if self.calls < 3:
                    raise RuntimeError(
                        "Invalid CDPR state after episode reset "
                        "(instruction=open_gripper, shell=None, reason=ee_outside_workspace)."
                    )
                return {"ok": True}

        module = types.SimpleNamespace(CDPRVisionLanguageEnv=FakeVisionEnv)
        _patch_training_reset_retries(module, max_attempts=3)
        env = module.CDPRVisionLanguageEnv()

        self.assertEqual(env.reset(options={"instruction_type": "open_gripper"}), {"ok": True})
        self.assertEqual(env.calls, 3)

    def test_training_reset_quarantines_cached_scene_and_uses_safe_final_start(self):
        class FakeRL:
            def __init__(self):
                self._current_wrapper_xml = Path("/tmp/bad-wrapper.xml")
                self._invalid_wrapper_paths = set()
                self._scene_name = "desk"
                self._rlvla_force_fresh_wrapper_on_next_reset = False

            @staticmethod
            def _default_ee_start():
                return np.asarray((0.12, -0.08, 0.40), dtype=np.float32)

            @staticmethod
            def _clamp_ee_target(value):
                return np.asarray(value, dtype=np.float32)

        class FakeVisionEnv:
            def __init__(self):
                self.env = FakeRL()
                self.calls = 0
                self.seen_options = []
                self._rlvla_compatible_wrapper_cache = {"desk": ["bad"]}

            def reset(self, options=None):
                self.calls += 1
                self.seen_options.append(dict(options or {}))
                if self.calls < 4:
                    raise RuntimeError(
                        "Invalid CDPR state after episode reset "
                        "(instruction=move_to_object, shell=None, reason=ee_outside_workspace)."
                    )
                return {"ok": True}

        module = types.SimpleNamespace(CDPRVisionLanguageEnv=FakeVisionEnv)
        _patch_training_reset_retries(module, max_attempts=4)
        env = module.CDPRVisionLanguageEnv()

        self.assertEqual(
            env.reset(options={"instruction_type": "move_to_object"}),
            {"ok": True},
        )
        self.assertEqual(env.calls, 4)
        self.assertIn(Path("/tmp/bad-wrapper.xml").resolve(), env.env._invalid_wrapper_paths)
        self.assertTrue(env.env._rlvla_force_fresh_wrapper_on_next_reset)
        self.assertEqual(env._rlvla_disabled_scene_cache_names, {"desk"})
        self.assertEqual(env._rlvla_compatible_wrapper_cache, {})
        np.testing.assert_allclose(
            env.seen_options[-1]["ee_start"],
            (0.0, 0.0, 0.40),
            atol=1e-7,
        )

    def test_training_reset_does_not_retry_unrelated_runtime_error(self):
        class FakeVisionEnv:
            def __init__(self):
                self.calls = 0

            def reset(self, options=None):
                del options
                self.calls += 1
                raise RuntimeError("programming defect")

        module = types.SimpleNamespace(CDPRVisionLanguageEnv=FakeVisionEnv)
        _patch_training_reset_retries(module, max_attempts=10)
        env = module.CDPRVisionLanguageEnv()

        with self.assertRaisesRegex(RuntimeError, "programming defect"):
            env.reset()
        self.assertEqual(env.calls, 1)

    def test_scene_cache_prebuild_progress_wraps_rl_wrapper_builds(self):
        class FakeRLEnv:
            calls: list[str] = []

            def _build_wrapper(self, scene, ee_start=None):
                del ee_start
                self.calls.append(str(scene.name))
                return Path("/tmp/fake_wrapper.xml")

        fake_rl_module = types.ModuleType("cdpr_dataset.rl_cdpr_env")
        fake_rl_module.CDPRLanguageRLEnv = FakeRLEnv

        class FakeVisionEnv:
            def _activate_scene_wrapper_cache(self, scene_wrapper_cache, texture_name_by_wrapper):
                del scene_wrapper_cache, texture_name_by_wrapper
                return {"activated": True}

        module = types.SimpleNamespace(CDPRVisionLanguageEnv=FakeVisionEnv)
        with mock.patch.dict(
            sys.modules,
            {
                "cdpr_dataset.rl_cdpr_env": fake_rl_module,
                "robots.cdpr.cdpr_dataset.rl_cdpr_env": fake_rl_module,
            },
        ), mock.patch.dict(os.environ, {"RANK": "1"}):
            _patch_scene_cache_prebuild_progress(
                module,
                ["--prebuild_scene_cache", "--scene_pool_size", "2", "--texture_pool_size", "3"],
            )

        env = FakeRLEnv()
        scene = types.SimpleNamespace(name="desk")

        with mock.patch.dict(os.environ, {"RANK": "1"}):
            self.assertEqual(env._build_wrapper(scene), Path("/tmp/fake_wrapper.xml"))
            self.assertEqual(FakeRLEnv.calls, ["desk"])
            self.assertTrue(getattr(FakeRLEnv._build_wrapper, "_rlvla_progress_wrapped", False))
            self.assertEqual(
                module.CDPRVisionLanguageEnv()._activate_scene_wrapper_cache({"desk": [Path("/tmp/a.xml")]}, {}),
                {"activated": True},
            )

    def test_split_wrapper_argv_strips_wrapper_only_options(self):
        external_script, forwarded, fast_args = _split_wrapper_argv(
            [
                "--external_grpo_script",
                "/tmp/external_grpo.py",
                "--tensorboard_rollout_every_global_steps",
                "100",
                "--tensorboard_metric_profile",
                "compact",
                "--rollout_image_size",
                "224",
                "--no-resume_actor_stats",
                "--first_stage_grpo_actor_stats_path",
                "/tmp/stage1_grpo_actor_stats.pt",
                "--stage2-grpo-actor-stats-path",
                "/tmp/stage2_grpo_actor_stats.pt",
                "--sparse-stage-init-log-std",
                "-0.7",
                "--ddp_timeout_seconds",
                "14400",
                "--ddp_rollout_sync_interval",
                "5",
                "--lr_scheduler",
                "cosine",
                "--lr_warmup_updates",
                "5",
                "--lr_min_factor",
                "0.25",
                "--max_train_reset_attempts",
                "7",
                "--rollout_steps",
                "170",
            ]
        )

        self.assertEqual(external_script, Path("/tmp/external_grpo.py").resolve())
        self.assertEqual(forwarded, ["--rollout_steps", "170"])
        self.assertEqual(fast_args.tensorboard_rollout_every_global_steps, 100)
        self.assertEqual(fast_args.tensorboard_metric_profile, "compact")
        self.assertEqual(fast_args.rollout_image_size, 224)
        self.assertFalse(fast_args.resume_actor_stats)
        self.assertEqual(fast_args.first_stage_grpo_actor_stats_path, Path("/tmp/stage1_grpo_actor_stats.pt"))
        self.assertEqual(fast_args.second_stage_grpo_actor_stats_path, Path("/tmp/stage2_grpo_actor_stats.pt"))
        self.assertAlmostEqual(fast_args.sparse_stage_init_log_std, -0.7)
        self.assertEqual(fast_args.ddp_timeout_seconds, 14400)
        self.assertEqual(fast_args.ddp_rollout_sync_interval, 5)
        self.assertEqual(fast_args.lr_scheduler, "cosine")
        self.assertEqual(fast_args.lr_warmup_updates, 5)
        self.assertAlmostEqual(fast_args.lr_min_factor, 0.25)
        self.assertEqual(fast_args.max_train_reset_attempts, 7)

    def test_patch_distributed_timeout_overrides_external_ppo_init(self):
        calls: list[tuple[str, float]] = []

        class FakeDist:
            @staticmethod
            def is_initialized():
                return False

            @staticmethod
            def init_process_group(*, backend, timeout):
                calls.append((backend, float(timeout.total_seconds())))

        fake_ppo = types.SimpleNamespace(
            dist=FakeDist,
            torch=types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True)),
        )
        module = types.SimpleNamespace(ppo=fake_ppo)

        _patch_distributed_timeout(module, timeout_seconds=14400)

        with mock.patch.dict(
            "os.environ",
            {"WORLD_SIZE": "2", "RANK": "0", "LOCAL_RANK": "1"},
            clear=False,
        ):
            self.assertEqual(module._rlvla_init_distributed(), (0, 1, 2))

        self.assertEqual(calls, [("nccl", 14400.0)])
        self.assertIs(fake_ppo._init_distributed, module._rlvla_init_distributed)

    def test_patch_distributed_timeout_wraps_direct_dist_init(self):
        calls: list[tuple[str, float]] = []

        class FakeDist:
            @staticmethod
            def init_process_group(*, backend, timeout):
                calls.append((backend, float(timeout.total_seconds())))

        original_init = lambda: (0, 0, 1)
        module = types.SimpleNamespace(dist=FakeDist, _init_distributed=original_init)

        _patch_distributed_timeout(module, timeout_seconds=14400)
        module.dist.init_process_group(backend="nccl")

        self.assertEqual(calls, [("nccl", 14400.0)])
        self.assertIs(module._rlvla_init_distributed, original_init)

    def test_ddp_sync_transform_routes_distributed_init_through_wrapper(self):
        source = (
            "def main():\n"
            "    rank, local_rank, world_size = ppo._init_distributed()\n"
            "    return rank, local_rank, world_size\n"
        )

        patched = _transform_external_grpo_source_for_ddp_sync(source)

        self.assertIn("rank, local_rank, world_size = _rlvla_init_distributed()", patched)
        self.assertNotIn("ppo._init_distributed()", patched)

    def test_ddp_sync_transform_adds_rank_barriers_before_update_and_train(self):
        source = (
            "        for update in range(1, args.total_updates + 1):\n"
            "            policy.eval()\n"
            "            do_rollout()\n"
            "            policy.train()\n"
        )

        patched = _transform_external_grpo_source_for_ddp_sync(source)

        self.assertIn('_rlvla_ddp_sync("pre_update", update=update, run_dir=run_dir)', patched)
        self.assertIn('_rlvla_ddp_sync("pre_train", update=update)', patched)

    def test_ddp_sync_transform_preserves_lchol_pre_update_hook(self):
        source = (
            "        for update in range(1, args.total_updates + 1):\n"
            "            _rlvla_lchol_pre_update(policy, args=args, update=update, run_dir=run_dir)\n"
            "            policy.eval()\n"
        )

        patched = _transform_external_grpo_source_for_ddp_sync(source)

        self.assertIn('_rlvla_ddp_sync("pre_update", update=update, run_dir=run_dir)', patched)
        self.assertIn("_rlvla_lchol_pre_update(policy, args=args, update=update, run_dir=run_dir)", patched)
        self.assertLess(
            patched.index('_rlvla_ddp_sync("pre_update"'),
            patched.index("_rlvla_lchol_pre_update"),
        )

    def test_ddp_sync_transform_adds_rollout_sync_and_update_marker(self):
        source = (
            "                if rollout_pbar is not None:\n"
            "                    rollout_pbar.update(1)\n\n"
            "            if rollout_pbar is not None:\n"
            "                rollout_pbar.close()\n"
            "            if is_main and update % args.save_every == 0:\n"
            "                save_checkpoint(\n"
            "                    run_dir=run_dir,\n"
            "                    step=global_step,\n"
            "                    vla=policy_core.vla,\n"
            "                    action_head=policy_core.action_head,\n"
            "                    log_std=policy_core.log_std,\n"
            "                )\n\n"
            "        if is_main:\n"
            "            save_checkpoint(\n"
        )

        patched = _transform_external_grpo_source_for_ddp_sync(source)

        self.assertIn("_rlvla_ddp_sync_rollout(update=update, rollout_step=rollout_step)", patched)
        self.assertIn("_rlvla_ddp_mark_update_complete(update=update, run_dir=run_dir)", patched)

    def test_ddp_sync_transform_disables_ddp_buffer_broadcasts(self):
        source = (
            "        ddp_kwargs = dict(\n"
            "            device_ids=[device.index],\n"
            "            find_unused_parameters=bool(args.ddp_find_unused_parameters),\n"
            "            gradient_as_bucket_view=True,\n"
            "        )\n"
        )

        patched = _transform_external_grpo_source_for_ddp_sync(source)

        self.assertIn("broadcast_buffers=False", patched)

    def test_ddp_sync_transform_removes_legacy_ppo_actor_stats_save(self):
        source = (
            "    actor_stats = {\"log_std\": log_std.detach().cpu()}\n"
            "    torch.save(actor_stats, ckpt_dir / \"grpo_actor_stats.pt\")\n"
            "    # Keep the familiar filename too so tooling can inspect log_std consistently.\n"
            "    torch.save(actor_stats, ckpt_dir / \"ppo_actor_stats.pt\")\n"
        )

        patched = _transform_external_grpo_source_for_ddp_sync(source)

        self.assertIn("grpo_actor_stats.pt", patched)
        self.assertNotIn("ppo_actor_stats.pt", patched)

    def test_ddp_sync_transform_skips_initial_full_model_sync_when_available(self):
        source = (
            "        ddp_params = inspect.signature(DDP.__init__).parameters\n"
            "        if \"static_graph\" in ddp_params:\n"
            "            ddp_kwargs[\"static_graph\"] = bool(args.ddp_static_graph)\n"
            "        policy = DDP(policy, **ddp_kwargs)\n"
        )

        patched = _transform_external_grpo_source_for_ddp_sync(source)

        self.assertIn('if "init_sync" in ddp_params:', patched)
        self.assertIn('ddp_kwargs["init_sync"] = False', patched)
        self.assertIn('rank={rank} entering DDP policy wrap', patched)
        self.assertIn('rank={rank} DDP policy ready', patched)

    def test_lr_scheduler_transform_applies_schedule_once_per_update_and_logs_lr(self):
        source = (
            "        for update in range(1, args.total_updates + 1):\n"
            "            _rlvla_ddp_sync(\"pre_update\", update=update, run_dir=run_dir)\n"
            "            policy.eval()\n"
            "                tb_writer.add_scalar(\"train/loss_total_mean\", avg_total_loss, global_step)\n"
        )

        patched = _transform_external_grpo_source_for_lr_scheduler(source)

        self.assertIn("_rlvla_apply_lr_schedule(optimizer, update=update, total_updates=args.total_updates)", patched)
        self.assertIn('tb_writer.add_scalar("train/learning_rate", _rlvla_current_lr(optimizer), global_step)', patched)

    def test_lr_scheduler_transform_preserves_lchol_pre_update_hook(self):
        source = (
            "        for update in range(1, args.total_updates + 1):\n"
            "            _rlvla_ddp_sync(\"pre_update\", update=update, run_dir=run_dir)\n"
            "            _rlvla_lchol_pre_update(policy, args=args, update=update, run_dir=run_dir)\n"
            "            policy.eval()\n"
            "                tb_writer.add_scalar(\"train/loss_total_mean\", avg_total_loss, global_step)\n"
        )

        patched = _transform_external_grpo_source_for_lr_scheduler(source)

        self.assertIn("_rlvla_lchol_pre_update(policy, args=args, update=update, run_dir=run_dir)", patched)
        self.assertIn("_rlvla_apply_lr_schedule(optimizer, update=update, total_updates=args.total_updates)", patched)

    def test_memory_safety_transform_guards_rollout_tap_records_and_logs_rss(self):
        external = Path("/Users/damirnurtdinov/Desktop/My Courses/Диплом/openvla-oft/vla-scripts/grpo_finetune_cdpr.py")
        if not external.exists():
            self.skipTest("Local OpenVLA-OFT GRPO script is not available.")

        patched = _transform_external_grpo_source_for_memory_safety(external.read_text(encoding="utf-8"))
        compile(patched, str(external), "exec")

        self.assertIn("record_rollout_tap = bool(", patched)
        self.assertIn("if record_rollout_tap:", patched)
        self.assertIn("_rlvla_log_memory(\"post_rollout\", update=update, is_main=is_main)", patched)
        self.assertIn("_rlvla_log_memory(\"post_train\", update=update, is_main=is_main)", patched)

    def test_cosine_lr_schedule_uses_warmup_then_decays_to_min_factor(self):
        self.assertAlmostEqual(
            _lr_schedule_factor(
                scheduler="cosine",
                update=1,
                total_updates=10,
                warmup_updates=2,
                min_factor=0.25,
            ),
            0.5,
        )
        self.assertAlmostEqual(
            _lr_schedule_factor(
                scheduler="cosine",
                update=10,
                total_updates=10,
                warmup_updates=2,
                min_factor=0.25,
            ),
            0.25,
        )

    def test_compact_tensorboard_profile_keeps_high_signal_tags_only(self):
        self.assertTrue(_tensorboard_tag_allowed("train/learning_rate"))
        self.assertTrue(_tensorboard_tag_allowed("stage/dense/mean_success"))
        self.assertTrue(_tensorboard_tag_allowed("stage/dense/mean_reward"))
        self.assertTrue(_tensorboard_tag_allowed("stage/dense/success_rate/catch_object"))
        self.assertTrue(_tensorboard_tag_allowed("stage/dense/reward/catch_object"))
        self.assertTrue(_tensorboard_tag_allowed("stage/dense/train/learning_rate"))
        self.assertTrue(_tensorboard_tag_allowed("rollout_episode/instruction_success_rate/pick_up"))
        self.assertTrue(_tensorboard_tag_allowed("rollout_episode/instruction_success_rate_mean"))
        self.assertTrue(_tensorboard_tag_allowed("rollout_episode/shell_success_rate/put_into_plate/shell_00"))
        self.assertTrue(_tensorboard_tag_allowed("lchol/replay/episodes_total"))
        self.assertTrue(
            _tensorboard_tag_allowed(
                "stage/sparse/buffer_episode_outcomes/global/cumulative/reward_1_ratio"
            )
        )
        self.assertTrue(_tensorboard_tag_allowed("lchol/reverse_frontier/shell_success_rate/put_into_plate/shell_00"))
        self.assertTrue(_tensorboard_tag_allowed("lchol/curriculum/reverse_frontier/put_into_plate/active_shell"))
        self.assertFalse(_tensorboard_tag_allowed("lchol/dense_gate/mean_success"))
        self.assertFalse(_tensorboard_tag_allowed("lchol/dense_stage/mean_success"))
        self.assertFalse(_tensorboard_tag_allowed("train/update_index"))
        self.assertFalse(_tensorboard_tag_allowed("stage/dense/train/update_index"))
        self.assertFalse(_tensorboard_tag_allowed("validation/action_x_hist", kind="histogram"))

    def test_lchol_transform_calls_after_update_before_stage_stop(self):
        external = Path("/Users/damirnurtdinov/Desktop/My Courses/Диплом/openvla-oft/vla-scripts/grpo_finetune_cdpr.py")
        if not external.exists():
            self.skipTest("Local OpenVLA-OFT GRPO script is not available.")

        patched = _transform_external_grpo_source_for_lchol(external.read_text(encoding="utf-8"))
        compile(patched, str(external), "exec")

        self.assertIn("_rlvla_lchol_after_update(", patched)
        self.assertIn("_rlvla_lchol_record_selected_step(", patched)
        self.assertIn("run_dir=run_dir,", patched)
        self.assertIn("_rlvla_lchol_should_stop_training(update=update)", patched)

    def test_tensorboard_writer_mirrors_scalars_by_lchol_stage(self):
        class FakeSummaryWriter:
            def __init__(self, *args, **kwargs):
                del args, kwargs
                self.scalars: list[tuple[str, float, int]] = []

            def add_scalar(self, tag, scalar_value, global_step=None, *args, **kwargs):
                del args, kwargs
                self.scalars.append((str(tag), float(scalar_value), int(global_step or 0)))
                return "scalar-result"

            def add_histogram(self, tag, values, *args, **kwargs):
                del tag, values, args, kwargs
                return "hist-result"

        class FakeRuntime:
            dense = True

            def dense_gate_active(self):
                return bool(self.dense)

        module = types.SimpleNamespace(SummaryWriter=FakeSummaryWriter, _rlvla_lchol_runtime=FakeRuntime())
        _patch_tensorboard_metric_filter(module, profile="compact")

        writer = module.SummaryWriter(log_dir="/tmp/tb")
        self.assertEqual(writer.add_scalar("train/reward_env_mean", 2.0, 10), "scalar-result")
        writer.add_scalar("train/update_index", 1.0, 10)
        writer.add_scalar("stage/dense/mean_success", 0.75, 10)
        module._rlvla_lchol_runtime.dense = False
        writer.add_scalar("validation/success_rate", 0.25, 20)

        tags = [tag for tag, _, _ in writer.scalars]
        self.assertIn("train/reward_env_mean", tags)
        self.assertIn("stage/dense/train/reward_env_mean", tags)
        self.assertIn("stage/dense/mean_success", tags)
        self.assertIn("validation/success_rate", tags)
        self.assertIn("stage/sparse/validation/success_rate", tags)
        self.assertNotIn("train/update_index", tags)
        self.assertNotIn("stage/dense/train/update_index", tags)

    def test_infer_resume_artifacts_prefers_grpo_actor_stats(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_dir = Path(tmp) / "step_0122400"
            adapter_dir = checkpoint_dir / "vla_cdpr_adapter"
            adapter_dir.mkdir(parents=True)
            (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
            (checkpoint_dir / "grpo_actor_stats.pt").write_text("grpo", encoding="utf-8")
            (checkpoint_dir / "ppo_actor_stats.pt").write_text("ppo", encoding="utf-8")

            artifacts = _infer_resume_artifacts(["--adapter_path", str(checkpoint_dir)])

            self.assertEqual(artifacts.checkpoint_dir, checkpoint_dir.resolve())
            self.assertEqual(artifacts.actor_stats_path, (checkpoint_dir / "grpo_actor_stats.pt").resolve())

    def test_infer_resume_artifacts_ignores_legacy_ppo_actor_stats(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_dir = Path(tmp) / "step_0122400"
            adapter_dir = checkpoint_dir / "vla_cdpr_adapter"
            adapter_dir.mkdir(parents=True)
            (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
            (checkpoint_dir / "ppo_actor_stats.pt").write_text("ppo", encoding="utf-8")

            artifacts = _infer_resume_artifacts(["--adapter_path", str(adapter_dir)])

            self.assertIsNone(artifacts.checkpoint_dir)
            self.assertIsNone(artifacts.actor_stats_path)

    def test_infer_resume_artifacts_uses_explicit_stage_grpo_actor_stats(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_dir = Path(tmp) / "step_0122400"
            checkpoint_dir.mkdir(parents=True)
            explicit_stats = checkpoint_dir / "grpo_actor_stats.pt"
            explicit_stats.write_text("grpo", encoding="utf-8")

            artifacts = _infer_resume_artifacts(
                ["--adapter_path", "/unused/checkpoint"],
                actor_stats_path=explicit_stats,
                require_actor_stats=True,
            )

            self.assertEqual(artifacts.checkpoint_dir, checkpoint_dir.resolve())
            self.assertEqual(artifacts.actor_stats_path, explicit_stats.resolve())

    def test_lchol_pre_update_resets_grpo_log_std_after_dense_gate(self):
        class FakeVisionEnv:
            def reset(self, options=None):
                return options

        def fake_validation(*args, **kwargs):
            del args, kwargs
            return {}

        module = types.SimpleNamespace(
            parse_args=lambda: types.SimpleNamespace(),
            CDPRVisionLanguageEnv=FakeVisionEnv,
            run_validation_rollouts=fake_validation,
        )
        _patch_lchol_runtime(module, _LCHOLWrapperArgs(enabled=True))

        class FakeRuntime:
            def __init__(self):
                self.synced = False
                self.pending = True

            def sync_dense_gate_state(self, *, run_dir):
                self.synced = str(run_dir)

            def consume_grpo_stats_reset_request(self):
                out = self.pending
                self.pending = False
                return out

        class FakeLogStd:
            device = "cpu"
            dtype = "float32"
            shape = (2,)

            def __init__(self):
                self.values = [9.0, 9.0]

            def fill_(self, value):
                self.values = [float(value), float(value)]

        runtime = FakeRuntime()
        policy = types.SimpleNamespace(log_std=FakeLogStd())
        optimizer = types.SimpleNamespace(state={policy.log_std: {"exp_avg": [1.0, 1.0]}})
        module._rlvla_lchol_runtime = runtime

        with tempfile.TemporaryDirectory() as tmp:
            module._rlvla_lchol_pre_update(
                policy,
                optimizer=optimizer,
                args=types.SimpleNamespace(init_log_std=-1.7),
                update=4,
                run_dir=Path(tmp),
            )
            events = [
                json.loads(line)
                for line in (Path(tmp) / "grpo_stage_transition.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]

        self.assertTrue(runtime.synced)
        self.assertEqual(policy.log_std.values, [-1.7, -1.7])
        self.assertNotIn(policy.log_std, optimizer.state)
        self.assertTrue(events[-1]["optimizer_state_cleared"])
        self.assertEqual(events[-1]["after_log_std"], [-1.7, -1.7])
        module._rlvla_lchol_pre_update(
            policy,
            optimizer=optimizer,
            args=types.SimpleNamespace(init_log_std=-0.3),
            update=5,
            run_dir=Path(tempfile.gettempdir()),
        )
        self.assertEqual(policy.log_std.values, [-1.7, -1.7])

    def test_lchol_pre_update_loads_second_stage_grpo_stats_after_dense_gate(self):
        class FakeVisionEnv:
            def reset(self, options=None):
                return options

        module = types.SimpleNamespace(
            parse_args=lambda: types.SimpleNamespace(),
            CDPRVisionLanguageEnv=FakeVisionEnv,
            run_validation_rollouts=lambda *args, **kwargs: {},
        )

        with tempfile.TemporaryDirectory() as tmp:
            stats_path = Path(tmp) / "grpo_actor_stats.pt"
            stats_path.write_text("placeholder", encoding="utf-8")

            _patch_lchol_runtime(
                module,
                _LCHOLWrapperArgs(enabled=True),
                fast_args=_FastWrapperArgs(second_stage_grpo_actor_stats_path=stats_path),
            )

            class FakeRuntime:
                def sync_dense_gate_state(self, *, run_dir):
                    del run_dir

                def consume_grpo_stats_reset_request(self):
                    return True

            class FakeTensor:
                device = "cpu"
                dtype = "float32"
                shape = (2,)

                def __init__(self, values):
                    self.values = [float(value) for value in values]

                def detach(self):
                    return self

                def reshape(self, *shape):
                    del shape
                    return self

                def numel(self):
                    return len(self.values)

                def to(self, *args, **kwargs):
                    del args, kwargs
                    return self

                def copy_(self, other):
                    self.values = list(other.values)

            class NoGrad:
                def __enter__(self):
                    return None

                def __exit__(self, exc_type, exc, tb):
                    del exc_type, exc, tb
                    return False

            fake_torch = types.SimpleNamespace(
                Tensor=FakeTensor,
                load=lambda path, map_location=None: {"log_std": FakeTensor([-0.4, -0.5])},
                no_grad=lambda: NoGrad(),
            )
            policy = types.SimpleNamespace(log_std=FakeTensor([9.0, 9.0]))
            module._rlvla_lchol_runtime = FakeRuntime()

            with mock.patch.object(grpo_fast, "torch", fake_torch):
                module._rlvla_lchol_pre_update(
                    policy,
                    args=types.SimpleNamespace(init_log_std=-1.7),
                    update=4,
                    run_dir=Path(tmp),
                )

            self.assertEqual(policy.log_std.values, [-0.4, -0.5])


if __name__ == "__main__":
    unittest.main()
