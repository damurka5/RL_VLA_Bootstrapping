# Methods B/D clarification — 29 October 2026

The former displayed action-prior dimension is now prose: eight slots with five channels give 40 values. Its equation label was unused elsewhere in the current manuscript.

Methods D now identifies the comparison explicitly. The historical CDPR path scored clipped commands and approximated persistent episode offsets through independent widened per-decision Gaussian marginals. The continuation records the pre-clipping sample and the realized offset used for that decision, and evaluates old/new conditional likelihoods on those same stored values. Source: `docs/reports/campaign/CDPR_ZERO_INIT_CORRECTION_PILOT.md`, §2 (relative to the code repository), and the associated implementation paths listed there. The promoted bounded-residual control is distinguished from the unpromoted correction-architecture candidate by the existing action-module audit.

Scoring original Gaussian samples before hard clipping is established policy-gradient practice, not a new method introduced by this paper. Fujita and Maeda (2018) discuss the conventional estimator alongside their distinct variance-reducing CAPG estimator: https://proceedings.mlr.press/v80/fujita18a.html . The manuscript cites this source for the sample/executed-action distinction; it does not claim to implement CAPG's tail-integrated estimator.

A concrete implementation reference is Stable Baselines3's `OnPolicyAlgorithm.collect_rollouts`: the Gaussian branch sends `clipped_actions` to the environment while storing the original `actions` and `log_probs` in the rollout buffer. https://stable-baselines3.readthedocs.io/en/master/_modules/stable_baselines3/common/on_policy_algorithm.html . Other policy parameterizations can handle action bounds differently; this comparison specifically concerns a Gaussian followed by hard clipping.

The revised prose introduces the old/new likelihood comparison, a scalar clipping example, and the persistent offset before the displayed sampling rule. It removes the phrase “key change” and does not attribute the final continuation endpoint solely to the likelihood correction, because optimizer restart and further training are also involved.

Backup: `archive/manuscript_20261029_before_methods_likelihood_clarification/`.
