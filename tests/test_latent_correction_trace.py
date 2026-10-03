"""Trace attribution reads direct components; it never relabels a correction.

``atanh(final) - prior`` is the bounded residual only for a legacy actor on a
deterministic arm. Correction-architecture traces carry the reference and
correction logits as separate arrays, and a correction trace without them is
refused rather than silently attributed to the residual.
"""

from __future__ import annotations

import unittest

import numpy as np

from tools.audit.summarize_kinematic_traces import _decision_pushes


def _trace(**extra):
    rng = np.random.default_rng(0)
    prior = rng.normal(0.5, 0.1, size=(3, 2, 4, 5)).astype(np.float32)
    residual = np.tanh(rng.normal(0.0, 2.0, size=prior.shape)).astype(np.float32)
    return prior, residual, {"decision_prior": prior, **extra}


class TracePushTests(unittest.TestCase):
    def test_legacy_trace_keeps_the_atanh_reconstruction(self):
        prior, residual, trace = _trace()
        trace["decision_final"] = np.tanh(prior + residual)
        push, correction = _decision_pushes(trace, 1)
        self.assertIsNone(correction)
        self.assertLess(float(np.abs(push - residual[:, 1]).max()), 1e-4)

    def test_correction_trace_uses_direct_components(self):
        prior, residual, trace = _trace()
        correction = np.full_like(prior, -2.5)
        reference_logit = prior + residual
        trace.update(
            policy_architecture=np.asarray("frozen_reference_logit_correction_v1"),
            decision_reference_logit=reference_logit,
            decision_correction_logit=correction,
            decision_final=np.tanh(reference_logit + correction),
        )
        push, corr = _decision_pushes(trace, 0)
        self.assertTrue(np.allclose(push, residual[:, 0], atol=1e-6))
        self.assertTrue(np.array_equal(corr, correction[:, 0]))
        # The reconstruction would have folded the correction into the push.
        rebuilt = np.arctanh(np.clip(trace["decision_final"][:, 0], -0.999999, 0.999999)) - prior[:, 0]
        self.assertGreater(float(np.abs(rebuilt - push).max()), 1.0)

    def test_correction_trace_without_components_is_refused(self):
        prior, residual, trace = _trace()
        trace.update(policy_architecture=np.asarray("frozen_reference_logit_correction_v1"),
                     decision_final=np.tanh(prior + residual))
        with self.assertRaises(ValueError):
            _decision_pushes(trace, 0)


if __name__ == "__main__":
    unittest.main()
