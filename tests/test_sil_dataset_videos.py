"""The GPU-free half of ``tools/audit/sil_dataset_videos.py``.

The selection is the whole tool: which episodes get filmed, and in what frame
order. Both failures are silent. Frames out of order look like a physics bug
rather than a sorting bug, and a sample taken off the front of the dataset shows
one rung of the ladder while claiming to show the mix.
"""

from __future__ import annotations

import unittest

import numpy as np

from tools.audit.sil_dataset_videos import (
    compose,
    group_episodes,
    pick_episodes,
    safe_name,
)


class GroupEpisodeTests(unittest.TestCase):
    def test_rows_come_back_in_decision_order(self) -> None:
        uid = np.array(["a", "a", "a"], dtype="U8")
        # Deliberately shuffled: nothing in the npz format promises order.
        decision = np.array([2, 0, 1])
        keep = np.array([True, True, True])
        self.assertEqual(
            group_episodes(uid, decision, keep)["a"], [1, 2, 0]
        )

    def test_unresolved_rows_are_excluded(self) -> None:
        uid = np.array(["a", "a", "b"], dtype="U8")
        decision = np.array([0, 1, 0])
        keep = np.array([True, False, True])
        grouped = group_episodes(uid, decision, keep)
        self.assertEqual(grouped["a"], [0])
        self.assertEqual(grouped["b"], [2])

    def test_an_episode_with_no_frames_disappears(self) -> None:
        uid = np.array(["a", "b"], dtype="U8")
        decision = np.array([0, 0])
        keep = np.array([False, True])
        self.assertEqual(set(group_episodes(uid, decision, keep)), {"b"})


class PickEpisodeTests(unittest.TestCase):
    def setUp(self) -> None:
        # 30 pick_up episodes then 5 bowl ones, which is how the dataset is
        # actually laid out: grouped by source recording.
        self.episodes = {f"p{i:02d}": [i] for i in range(30)}
        self.episodes.update({f"b{i:02d}": [i] for i in range(5)})
        self.instruction_of = {
            uid: ("pick_up" if uid.startswith("p") else "put_into_bowl")
            for uid in self.episodes
        }

    def test_each_instruction_gets_its_own_quota(self) -> None:
        chosen = pick_episodes(
            self.episodes, self.instruction_of, per_instruction=10, seed=0
        )
        self.assertEqual(len(chosen["pick_up"]), 10)
        self.assertEqual(sorted(chosen), ["pick_up", "put_into_bowl"])

    def test_a_short_instruction_gives_what_it_has(self) -> None:
        chosen = pick_episodes(
            self.episodes, self.instruction_of, per_instruction=10, seed=0
        )
        self.assertEqual(len(chosen["put_into_bowl"]), 5)

    def test_the_sample_is_not_the_front_of_the_dataset(self) -> None:
        # Taking the first ten would show one recording at one rung.
        chosen = pick_episodes(
            self.episodes, self.instruction_of, per_instruction=10, seed=0
        )
        self.assertNotEqual(
            chosen["pick_up"], [f"p{i:02d}" for i in range(10)]
        )

    def test_selection_is_reproducible_and_seed_dependent(self) -> None:
        a = pick_episodes(
            self.episodes, self.instruction_of, per_instruction=5, seed=7
        )
        b = pick_episodes(
            self.episodes, self.instruction_of, per_instruction=5, seed=7
        )
        c = pick_episodes(
            self.episodes, self.instruction_of, per_instruction=5, seed=8
        )
        self.assertEqual(a, b)
        self.assertNotEqual(a["pick_up"], c["pick_up"])

    def test_no_episode_is_filmed_twice(self) -> None:
        chosen = pick_episodes(
            self.episodes, self.instruction_of, per_instruction=10, seed=3
        )
        for uids in chosen.values():
            self.assertEqual(len(set(uids)), len(uids))


class ComposeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.overview = np.zeros((4, 6, 3), dtype=np.uint8)
        self.wrist = np.ones((4, 6, 3), dtype=np.uint8)

    def test_both_places_the_cameras_side_by_side(self) -> None:
        frame = compose(self.overview, self.wrist, "both")
        self.assertEqual(frame.shape, (4, 12, 3))
        self.assertTrue((frame[:, :6] == 0).all())
        self.assertTrue((frame[:, 6:] == 1).all())

    def test_single_camera_is_passed_through_unchanged(self) -> None:
        self.assertTrue(
            (compose(self.overview, self.wrist, "wrist") == 1).all()
        )
        self.assertEqual(
            compose(self.overview, self.wrist, "overview").shape, (4, 6, 3)
        )


class SafeNameTests(unittest.TestCase):
    def test_the_uid_survives_as_a_filename(self) -> None:
        # The uid carries the rung and the world, which is what makes a video
        # traceable back to its recording.
        self.assertEqual(
            safe_name("p2_demos/replay_p2_0.20_record_01/r1w0742"),
            "p2_demos__replay_p2_0.20_record_01__r1w0742",
        )


if __name__ == "__main__":
    unittest.main()
