"""Final-prompt labels preserve source actions/frames and isolate holdout episodes."""

import unittest

import numpy as np

from tools.audit.build_widowx200_composition_sft import KINDS, build
from tools.audit.sil_refresh_priors import check_final_prompt
from tools.audit.sil_sft import BalancedRowSampler, provided_split, resolve_frame_rows, select_frame_budget
from rl_vla_bootstrapping.simulation.cdpr_batched_tasks import INSTRUCTION_TO_ID


def source_dataset(episodes=20):
    rows = []
    for kind in KINDS:
        for world in range(episodes):
            obj = "apple" if world % 2 else "tomato"
            text = {"move_to_object": f"move to {obj}", "pick_up": f"pick up {obj}"}.get(
                kind, f"put {obj} into {kind.removeprefix('put_into_')}")
            for decision in range(2):
                rows.append((kind, world, text, decision))
    n = len(rows)
    return {
        "state": np.arange(n, dtype=np.float32)[:, None],
        "prior": np.arange(n, dtype=np.float32)[:, None, None],
        "action": np.arange(n, dtype=np.float32)[:, None, None],
        "action_mask": np.ones((n, 1), bool),
        "instruction_id": np.array([INSTRUCTION_TO_ID[r[0]] for r in rows]),
        "instruction_text": np.array([r[2] for r in rows]),
        "episode_uid": np.array([f"source_{r[0]}/r0w{r[1]}" for r in rows]),
        "decision_index": np.array([r[3] for r in rows]),
    }


class CompositionDatasetTests(unittest.TestCase):
    def test_labels_and_masked_actions_stay_attached_to_real_source_frames(self):
        source = source_dataset()
        data, report = build(source, rows_per_cell=12)
        original_rows = data["state"][:, 0].astype(int)
        for column in ("state", "prior", "action", "action_mask", "episode_uid", "decision_index"):
            np.testing.assert_array_equal(data[column], source[column][original_rows])
        frames = {f"source_{kind}": {"source_key": f"source_{kind}", "decisions": 2,
                                    "world_column": {i: i for i in range(20)}} for kind in KINDS}
        found, _ = resolve_frame_rows(data["episode_uid"], data["decision_index"], frames)
        self.assertTrue(found.all())
        final = data["sft_view"] == "final_prompt"
        original = ~final
        self.assertTrue(all(str(text).startswith("put ") for text in data["instruction_text"][final]))
        np.testing.assert_array_equal(data["instruction_text"][original], data["teacher_instruction_text"][original])
        for uid in np.unique(data["episode_uid"]):
            mask = data["episode_uid"] == uid
            self.assertEqual(np.unique(data["instruction_text"][mask]).size, 1)
            self.assertEqual(np.unique(data["sft_view"][mask]).size, 1)
        bowl_teacher = data["teacher_instruction_id"] == INSTRUCTION_TO_ID["put_into_bowl"]
        self.assertTrue((data["destination"][bowl_teacher] == "bowl").all())
        self.assertFalse(data["full_chain_success"].any())
        self.assertTrue(report["priors_stale"])
        self.assertFalse(report["continuous_full_task_demonstrations"])
        self.assertEqual(check_final_prompt(data, prefix="")["violations"], [])

    def test_every_sampling_cell_has_whole_episodes_on_both_sides(self):
        data, report = build(source_dataset(), rows_per_cell=1)
        train, val = provided_split(data)
        self.assertFalse(set(data["episode_uid"][train]) & set(data["episode_uid"][val]))
        for cell in report["cells"].values():
            self.assertGreater(cell["train_decisions"], 0)
            self.assertGreater(cell["validation_decisions"], 0)
        # A one-row quota still keeps two complete two-decision episodes.
        self.assertTrue(all(cell["decisions"] == 4 for cell in report["cells"].values()))

    def test_original_instruction_exposure_is_preserved_in_the_shared_sampler(self):
        data, _ = build(source_dataset())
        train, _ = provided_split(data)
        sampler = BalancedRowSampler(data, train, seed=7)
        drawn = sampler.draw(20000)
        share = (data["sft_view"][drawn] == "original_retention").mean()
        self.assertAlmostEqual(share, .2, delta=.015)
        self.assertEqual(sampler.report()["destination_column"], "sampling_destination")
        final = data["sft_view"][drawn] == "final_prompt"
        for destination in ("plate", "bowl"):
            for stage in ("move_to", "pick_up", "placement"):
                self.assertTrue((final & (data["destination"][drawn] == destination) &
                                 (data["stage_name"][drawn] == stage)).any())

    def test_frame_budget_covers_both_split_sides_and_every_sampling_cell(self):
        data, _ = build(source_dataset())
        strata = np.array([f"{v}/{d}/{s}" for v, d, s in zip(
            data["sft_validation"], data["sampling_destination"], data["stage_name"])])
        chosen = select_frame_budget(np.ones(len(strata), bool), budget=20, seed=0, strata=strata)
        self.assertEqual(len(chosen), 20)
        self.assertEqual(set(strata[chosen]), set(strata))
        with self.assertRaisesRegex(ValueError, "cannot cover"):
            select_frame_budget(np.ones(len(strata), bool), budget=10, seed=0, strata=strata)

    def test_missing_pickup_material_cannot_silently_turn_into_placement_only_sft(self):
        source = source_dataset()
        keep = source["instruction_id"] != INSTRUCTION_TO_ID["pick_up"]
        with self.assertRaisesRegex(ValueError, "pick_up: need at least"):
            build({key: value[keep] for key, value in source.items()})

    def test_provided_holdout_rejects_episode_leakage(self):
        data, _ = build(source_dataset())
        uid = data["episode_uid"][0]
        rows = np.flatnonzero(data["episode_uid"] == uid)
        data["sft_validation"][rows[0]] = ~data["sft_validation"][rows[1]]
        with self.assertRaisesRegex(ValueError, "leaks episode_uid"):
            provided_split(data)

    def test_non_graspable_reach_targets_are_excluded_and_build_is_reproducible(self):
        source = source_dataset()
        source["instruction_text"][:2] = "move to plate"
        first, report = build(source, seed=9)
        second, _ = build(source, seed=9)
        self.assertEqual(report["dropped_non_graspable_target_episodes"], {"plate": 1})
        for column in first:
            np.testing.assert_array_equal(first[column], second[column])

    def test_grpo_scene_copies_share_their_view_destination_and_holdout_side(self):
        source = source_dataset()
        scenes = []
        for index, uid in enumerate(source["episode_uid"]):
            stem, tail = str(uid).rsplit("w", 1)
            scene = int(tail) // 2
            scenes.append(f"{stem}s{scene}")
            obj = "apple" if scene % 2 else "tomato"
            old_obj = "apple" if int(tail) % 2 else "tomato"
            source["instruction_text"][index] = str(source["instruction_text"][index]).replace(old_obj, obj)
        source["scene_uid"] = np.array(scenes)
        data, report = build(source, rows_per_cell=8)
        train, val = provided_split(data)
        self.assertFalse(set(data["scene_uid"][train]) & set(data["scene_uid"][val]))
        for scene in np.unique(data["scene_uid"]):
            mask = data["scene_uid"] == scene
            self.assertEqual(np.unique(data["sft_view"][mask]).size, 1)
            self.assertEqual(np.unique(data["sampling_destination"][mask]).size, 1)
            self.assertEqual(np.unique(data["sft_validation"][mask]).size, 1)
        self.assertEqual(report["split_group"], "recorded scene_uid")


if __name__ == "__main__":
    unittest.main()
