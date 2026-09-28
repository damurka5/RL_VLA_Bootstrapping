# PPO Analysis for OpenVLA CDPR Fine-Tuning

## Abstract

This report summarizes the current PPO formulation used in the OpenVLA-based CDPR training pipeline, clarifies the roles of the action head, the value head, and the exploration parameters, and records the current estimate of the trainable parameter count. The analysis is based on the current implementations in `openvla-oft/vla-scripts/ppo_finetune_cdpr.py`, `openvla-oft/prismatic/models/action_heads.py`, and the deterministic runtime paths in `rl_vla_bootstrapping/cli/run_cdpr_policy.py` and `rl_vla_bootstrapping/cli/validate_cdpr_policy.py`.

## 1. Policy parameterization

At training time, the policy receives a multimodal observation consisting of the primary camera image, the wrist camera image when enabled, and the text instruction. The policy does not receive the current action as an explicit input, and it does not receive the previous action as a dedicated state variable. The policy must therefore infer the robot configuration from visual context and language conditioning alone.

The action head produces a pre-squash vector `u` in `R^d`, where `d = ACTION_DIM * NUM_ACTIONS_CHUNK`. In the current CDPR configuration, `ACTION_DIM = 5` and `NUM_ACTIONS_CHUNK = 8`, hence `d = 40`. The deterministic action corresponding to this output is `tanh(u)`, which lies componentwise in `[-1, 1]`.

The stochastic PPO policy is defined in a latent pre-`tanh` space. Let `eps ~ N(0, I)` be standard Gaussian noise and let `sigma = exp(log_std)` be the learned exploration scale. The sampled latent action is

`z = u + sigma * eps`

and the executed normalized action is

`a = tanh(z)`.

This construction ensures that the executed action remains bounded in `[-1, 1]` while exploration is still defined through a Gaussian distribution in the unconstrained latent space.

## 2. Interpretation of `sigma`, `eps`, and `log_std`

The variable `eps` is not learned. It is random standard-normal noise drawn independently at action-sampling time. The variable `sigma` is learned. In the current implementation it is not predicted by the action head from the observation; instead it is obtained from a trainable parameter vector `log_std`, and `sigma = exp(log_std)` guarantees positivity.

Consequently, the action head itself is responsible only for the mean behavior of the policy, namely the vector `u`. Exploration is controlled by a separate learned object, the `log_std` vector. The full PPO actor is therefore the pair `(u, sigma)` rather than the action head alone.

Because `POLICY_ACTION_DIM = 40` in the current 5-DoF, chunk-8 setting, `log_std` contains exactly 40 trainable scalar parameters. These 40 values correspond to the 40 scalar components of the flattened action chunk, that is, one exploration scale for each scalar output across all 8 low-level actions in the chunk.

## 3. Role of the value head

The value head does not evaluate a sampled action in the sense of a `Q(s, a)` critic. Instead, it predicts a state value `V(s)`, where the state is the current multimodal observation. Its purpose is to provide a baseline used in return and advantage estimation.

For this reason, the following interpretation is inaccurate: the value head does not directly tell the action head that the sampled noisy action is better than the unsampled mean action. Instead, PPO operates as follows. The policy samples an action from its current distribution, the environment returns reward, and an advantage estimate is computed from the realized return relative to the value baseline. If the sampled action produced a positive advantage, PPO increases its probability under the policy. If it produced a negative advantage, PPO decreases its probability.

Thus, the actor is trained through the probability of sampled actions, while the value head is trained to predict expected returns from observations.

## 4. How sampled actions are reinforced

The mechanism of reinforcement is probabilistic. During rollout, the trainer samples an action from the current policy distribution in the latent pre-`tanh` space and executes the squashed bounded version in the environment. The trainer stores the sampled action and its log-probability under the old policy. During the PPO update, the same sampled action is evaluated under the new policy, and the ratio between new and old probabilities is formed. This ratio is then multiplied by the estimated advantage.

The practical meaning is simple. If a sampled action led to an outcome better than expected, PPO changes the parameters so that actions of this kind become more likely in the future under similar observations. If a sampled action led to an outcome worse than expected, PPO changes the parameters so that actions of this kind become less likely.

Therefore, the actor learns from reward-weighted probability updates, not from a direct supervised comparison between the deterministic mean action and the noisy sampled action.

## 5. Training, validation, and inference

Training is stochastic. The rollout path samples actions using the squashed Gaussian rule `a = tanh(u + sigma * eps)`. Although the rollout loop sets the module to evaluation mode in the PyTorch sense, exploration is still present because the noise term is sampled explicitly in code.

Validation inside the PPO trainer is deterministic. It uses the mean action `tanh(u)` and does not add exploration noise. The same is true for the standard inference script `rl_vla_bootstrapping/cli/run_cdpr_policy.py`, which computes `pred_pre = action_head.predict_action(...)` and then returns `tanh(pred_pre)`. The deterministic validator in `rl_vla_bootstrapping/cli/validate_cdpr_policy.py` follows the same action-prediction path.

This distinction is important. The PPO actor statistics stored in `ppo_actor_stats.pt` affect training-time exploration because they contain the learned `log_std`. They are not used by the normal deterministic inference runner. As a consequence, pathological behavior at inference time is evidence of large or poorly calibrated mean outputs `u`, not direct evidence of large exploration noise.

## 6. Correction of the previous action-sampling bug

In an earlier version of the PPO trainer, the mean action was first bounded by `tanh`, then an unconstrained Gaussian sample was taken around that bounded mean, and only afterwards the action was clipped before being sent to the environment. This created a mismatch: PPO optimized the log-probability of an unclipped Gaussian sample, whereas the environment executed a clipped action. The executed action and the optimized action were not the same object.

The current implementation fixes this by sampling in the latent pre-`tanh` space and then applying `tanh` to the sample itself. This makes the executed action, the stored action, and the action whose log-probability is optimized consistent with one another. The correction does not eliminate saturation by itself, but it removes a major source of biased credit assignment near the action limits.

## 7. Trainable parameter count

The current PPO implementation trains four groups of parameters: LoRA parameters inside the OpenVLA backbone, the action head, the value head, and the 40-dimensional `log_std` vector.

For the present 5-DoF, chunk-8 configuration, the exact counts of the non-LoRA trainable components are as follows. The action head contains `117,538,821` trainable parameters. The value head contains `83,935,233` trainable parameters. The `log_std` vector contributes `40` parameters. Hence, even before counting any LoRA weights, the PPO setup already trains `201,474,094` parameters.

The LoRA count depends on the precise backbone structure matched by the target module list. The model `openvla/openvla-7b` corresponds to the DINO-SigLIP 224px OpenVLA family derived from `prism-dinosiglip-224px+7b`. Because the target list includes LLM projection modules as well as vision transformer modules and projector layers, the LoRA parameter count is substantial. A reasonable estimate for the current configuration is approximately `107.9M` trainable LoRA parameters. Under that estimate, the full PPO trainable total is approximately `309.4M` parameters.

The precise LoRA count for a specific run should be taken from the startup print emitted by the trainer, which reports the quantities `lora`, `action_head`, and `value_head` directly.

## 8. Meaning of `resume_actor_stats: false`

The configuration flag `resume_actor_stats: false` does not affect the adapter weights, the action head, or the value head. It only prevents the trainer from restoring the saved PPO actor statistics file containing `log_std`. In practical terms, this means that the policy resumes from the learned mean behavior encoded in the adapter and action head, while the exploration scales are reset to the configured initialization `init_log_std`.

For the current CDPR setup, this reset applies to the 40 global exploration parameters corresponding to the flattened 8-by-5 action chunk.

## 9. Practical conclusion

The current PPO actor should be understood as a stochastic distribution over bounded action chunks rather than as a deterministic action head perturbed by an external heuristic. The action head provides the mean structure of behavior, the `log_std` vector controls exploration amplitude, the value head provides the baseline required for advantage estimation, and PPO reinforces sampled actions by increasing or decreasing their probability according to the sign and magnitude of the estimated advantage. Deterministic inference bypasses the stochastic exploration component and executes `tanh(u)` directly.
