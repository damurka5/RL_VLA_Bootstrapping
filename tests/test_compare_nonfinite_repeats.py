"""Paired simulator health must not perturb the pre-existing promotion statistics."""
import copy
import unittest

import numpy as np

from tools.audit.compare_put_into_repeats import compare_pair


class NonfiniteComparisonTests(unittest.TestCase):
    def fixture(self):
        scenes = [f"s{i}" for i in range(40)]
        def evaluations(candidate):
            return [{"episodes": {uid: {
                "destination": "plate" if i % 2 else "bowl", "target_catalog": "apple",
                "strict": i < 10 + 5 * candidate + rep,
                "native": i < 20, "grasped": i < 30, "lifted": i < 25,
                "carry_slip": i >= 25, "wrong_place": i >= 20,
                "non_finite": bool(candidate and i >= 30),
            } for i, uid in enumerate(scenes)}} for rep in range(4)]
        return scenes, evaluations(False), evaluations(True)

    def test_reports_paired_failure_increase(self):
        scenes, base, cand = self.fixture()
        result = compare_pair(base, cand, scenes, rng=np.random.default_rng(0), resamples=1000)
        metric = result["metrics"]["non_finite"]
        self.assertEqual(metric["baseline"], 0)
        self.assertEqual(metric["candidate"], 0.25)
        self.assertEqual(metric["difference"], 0.25)
        self.assertGreater(metric["ci95"][0], 0)
        self.assertLess(metric["permutation_p"], 0.05)

    def test_missing_flags_are_not_treated_as_zero_failures(self):
        scenes, base, cand = self.fixture()
        del cand[1]["episodes"][scenes[0]]["non_finite"]
        result = compare_pair(base, cand, scenes, rng=np.random.default_rng(0), resamples=100)
        self.assertNotIn("non_finite", result["metrics"])
        self.assertIn("non_finite", result["unavailable_metrics"])

    def test_new_metric_preserves_other_statistics_and_shared_rng(self):
        scenes, base, cand = self.fixture()
        old = copy.deepcopy(cand)
        for ev in old:
            for episode in ev["episodes"].values():
                del episode["non_finite"]
        old_rng, new_rng = np.random.default_rng(7), np.random.default_rng(7)
        old_result = compare_pair(base, old, scenes, rng=old_rng, resamples=100)
        new_result = compare_pair(base, cand, scenes, rng=new_rng, resamples=100)
        new_result["metrics"].pop("non_finite")
        old_result.pop("unavailable_metrics")
        self.assertEqual(old_result, new_result)
        # The caller reuses this RNG for the next checkpoint comparison.
        np.testing.assert_array_equal(old_rng.random(30), new_rng.random(30))


if __name__ == "__main__":
    unittest.main()
