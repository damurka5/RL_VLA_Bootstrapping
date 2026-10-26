# Mathematical foundation for the RA-L paper

**Research working note, 11 October 2026; manuscript status updated 21 October.** Proofs and assumptions supporting the working manuscript. The article is now prose and includes recorded 40M development results (61.43%). This note does not change the algorithm or launch experiments; its conditional finite-model results do not certify the neural checkpoint or hardware transfer. Historical references below to an unavailable >50% checkpoint describe the earlier evidence state.

The paper should explain three mechanisms: **a correct probability model for the sampled actions; reliable intermediate credit; and conditions connecting a parameter update to complete-task success.** Supporting mathematics should predict measurements and possible failures. A generic policy-gradient theorem alone is insufficient.

## 1. Mathematical object and implementation contract

Fix a distribution of scenes/instructions c, horizon H, and binary strict outcome Y. Define

$$J(\theta)=\mathbb E_{c,\tau\sim P_\theta(\cdot\mid c)}Y(\tau).$$

History h includes controller/chunk memory, persistent milestone latches and remaining time. The actor can use its actual observation features x(h), without access to all this mathematical state. Treat a numerical simulator failure as an absorbing failure with Y=0 for the all-attempts objective.

The frozen SmolVLA prior p and its sampling randomness are part of the conditioning information. The bounded residual mean is

$$\mu_\theta(x,p)=\tanh\{p+s_r\tanh f_\theta(x,p)\}.$$

An exogenous offset B is drawn once per episode; the predictable gate g_t(h_t) gives b_t=g_t B. Conditional latent actions and execution are

$$u_t\sim\mathcal N(\mu_\theta(x_t,p_t)+b_t,\Sigma_\theta),\qquad a_t=C(u_t)=\operatorname{clip}(u_t,-1,1).$$

Index t by independent sampled slots/decisions, not by repeated physics substeps. For independent diagonal-Gaussian chunk slots, unused slots can be integrated out; include a slot's score only when its execution is determined before seeing that slot's sample. Terminal masks must be predictable in that sense. Stage-based retrospective filtering is a separate issue.

**Training versus deployment:** this J refers to the specified stochastic policy. The recorded evaluator uses deterministic residual means and a stochastic VLA prior, while training adds residual Gaussian exploration and gated offsets. A training-gradient result is not automatically a result for deterministic-mean deployment. Keep J_explore and J_deploy separate; measure both or explicitly bound their difference. A Gaussian-to-point-mass TV bound is generally vacuous.

### Actual stage credit

Let M1, M2, M3=Y be the approach, pickup and placement returns. For each scene group, define A_k by centering its binary returns and dividing by population standard deviation floored at epsilon; apply configured advantage clipping. Masks exclude invalid candidates and uninformative groups.

For a row in stage k<3, the implementation adds full-task credit:

$$\widetilde A_{ik}=U_{ik}A_{ik}+\lambda U_{i3}T_\kappa(A_{i3},M_{ik}),$$

$$T_\kappa(a,m)=\begin{cases}\kappa a&m=1,\ a<0,\\a&\text{otherwise}.\end{cases}$$

Here U denotes stage/group usability. For placement, the credit is U_i3 A_i3, without doubling the terminal bonus. The current local YAML sets lambda=1 and kappa=0; the unavailable latest run's resolved configuration must confirm these values. This is protection of an achieved milestone from *negative terminal bonus*, not a global retention guarantee.

On a fixed batch, candidate- and stage-balanced row weights are proportional to

$$w_{ikt}=\frac{m_k}{\sum_{j\in K_B}m_j}\frac1{n_k T_{ik}}.$$

Only retained candidates/rows count in n_k and T_ik. The global distributed implementation adds the factors needed to undo DDP rank averaging and scale minibatch sampling. The policy surrogate is

$$S_B(\theta)=\sum_{ikt}w_{ikt}\min\{r_{it}\widetilde A_{ik},\operatorname{clip}(r_{it},1-\epsilon_-,1+\epsilon_+)\widetilde A_{ik}\},$$

with per-slot conditional latent likelihood ratio r. There are also entropy and mean-action penalties, Adam updates, gradient clipping and repeated minibatches. This is the empirical surrogate to describe, not simply E[M1+M2+M3].

## 2. Proposition A: exact latent score with a persistent offset

**Assumptions.** Fixed horizon (or absorbing padding); parameter-independent environment/controller, reset law, offset law and frozen-prior mechanism; a differentiable positive latent density with integrable score; differentiation under the expectation is valid. Gates are parameter-independent functions of the recorded augmented history. If prior/LoRA parameters are trained, their scores cannot silently be omitted.

For the augmented trajectory z=(B,h_0,u_0,...),

$$P_\theta(z\mid c)=q(B\mid c)\rho_0(h_0\mid c)\prod_{t=0}^{H-1}\pi_\theta(u_t\mid h_t,p_t,b_t)K(h_{t+1}\mid h_t,C(u_t),c).$$

Frozen-prior factors can be included in K or factored explicitly. Therefore

$$\nabla J=\mathbb E\left[Y\sum_t\nabla\log\pi_\theta(u_t\mid h_t,p_t,b_t)\right].$$

**Proof.** Differentiate the normalized trajectory density. All factors except the latent policy are parameter-independent at fixed recorded history and latent sample. The likelihood-ratio identity gives the expression. Dependence of later histories on earlier actions is already accounted for by the trajectory distribution; do not differentiate through the simulator again in this score estimator. The offset density cancels between old and new joint trajectory likelihoods. The conditional scores must use the same recorded B/gate values.

For fixed covariance, the mean contribution is

$$\nabla\log\pi_\theta=(\nabla\mu_\theta)^\top\Sigma^{-1}(u-\mu_\theta-b).$$

If covariance is learned, include its Gaussian score too. The product of per-slot ratios is the trajectory ratio; the implemented individual-slot PPO ratios are a surrogate and do not equal that product away from the behavior policy.

**Implication.** Clipping does not invalidate the score of the *unclipped sample*. Scoring C(u) as though it were sampled from a continuous Gaussian instead assigns the wrong likelihood to boundary atoms. Multiplying independent one-step marginals for a shared B also fails to represent its temporal dependence. Conditioning on B preserves credit through the returns and visited histories.

**Limit.** Correct scores do not repair stage selection, reward changes, normalization, stale-policy PPO updates, or exclusion of failed episodes. If failures are excluded from training statistics, the update is not the exact score estimator of the all-attempts J above.

**Prior art.** This is an application of the score-function identity. Bounded-action estimator analysis is established in [Fujita and Maeda, CAPG, ICML 2018](https://proceedings.mlr.press/v80/fujita18a.html). Their clipping-aware estimator is distinct from this retained-latent estimator and can reduce variance. Do not claim unbiasedness implies minimum variance or that latent scoring is a new discovery.

## 3. Proposition B: why sparse milestones produce usable groups

For G conditionally independent trajectories of one fixed scene with binary milestone probability p,

$$P_{\rm mixed}(p,G)=1-p^G-(1-p)^G.$$

**Proof.** The only groups with zero binary variance are all successes and all failures. Their probabilities are p^G and (1-p)^G.

For G=8, p=0.03 yields 21.626% mixed groups; p=0.30 yields 94.229%. Thus an intermediate milestone whose probability is closer to one half can provide contrast before final success is common. A milestone near p=1 supplies little contrast; merely being easier does not always help.

If groups are independent, the probability of obtaining at least one mixed group in N attempts is 1-(1-P_mixed)^N, and the expected waiting time is 1/P_mixed. These quantify signal availability, not variance reduction, policy improvement, or total sample complexity.

Use E_c[P_mixed(p(c),G)] across scenes, not P_mixed(E_c p(c),G). Shared scenes do not by themselves violate conditional independence, but correlated rollout randomness or common offsets across candidates can. Group members need independent episode offsets conditional on c for this binomial calculation. Additional usability/validity filters can remove mixed groups.

### Binary group normalization: a useful exact benchmark

For fixed c, let p=Pr(Y=1|c), score Z_i=grad log P_theta(tau_i|c), Q=K/G, K~Binomial(G,p), d(Q)=max(sqrt(Q(1-Q)),epsilon), and A_i=(Y_i-Q)/d(Q). Without advantage clipping, trajectory-length normalization or filtering,

$$\mathbb E\left[\frac1G\sum_i A_i Z_i\mid c\right]=c_{G,\epsilon}(p)\nabla p,$$

$$c_{G,\epsilon}(p)=\frac{\mathbb E[Q(1-Q)/d(Q)]}{p(1-p)}>0\quad(0<p<1).$$

**Proof.** The score identity gives E[Z|Y=1]=grad p/p and E[Z|Y=0]=-grad p/(1-p). Conditional on the group labels, the success contribution is Q(1-Q)grad p/[d(Q)p], and the failure contribution is Q(1-Q)grad p/[d(Q)(1-p)]. Sum and average over K. Constant groups contribute zero. With no variance normalization, this recovers (G-1)/G times grad p. With G=2 and inactive epsilon flooring, c=1.

The expected direction for **one scene** is positive-scaled ascent. Shared parameters across scenes instead optimize a transformed utility E_c F_G(p_c), F'_G=c_G; they need not ascend the arithmetic-mean success J. Restriction to stage rows, candidate-length averaging, asymmetric protection, invalid filtering and regularization prevent substituting the implemented gradient into this identity. At the first on-policy point PPO clipping is inactive locally; later minibatch updates need separate analysis.

This binary-reward analysis overlaps existing work: [Mroueh, ICML 2026](https://proceedings.mlr.press/v306/mroueh26a.html) and [Bay and Yearick, 2026 preprint](https://arxiv.org/abs/2607.00152). Use it as supporting mathematics with attribution, not a novelty claim.

## 4. Proposition C: an exact improvement result for protected stage credit

This is the most direct controlled explanation of the project's **specific preservation rule**. It is a benchmark model, not a proof about every neural CDPR policy.

**Model.** Three sequential Bernoulli gates. Stage k is reached only after earlier gates succeed. At a reached stage sample X_k~Bernoulli(q_k), q_k=sigmoid(theta_k), with distinct scalar parameter theta_k for each stage. A failed gate ends the episode. M_k=product_{j<=k} X_j and Y=M_3. Hence J=q1 q2 q3. Groups contain independent trajectories of the same task. Each reached stage emits one scored decision. Use current-policy scores, nonnegative row/stage weights, sign-preserving binary group normalization/clipping, lambda>=0, and the actual protected bonus with kappa=0. Omit entropy/action penalties and optimizer momentum.

**Statement.** Every realized batch's raw policy-ascent direction has d_k>=0. Thus for every finite eta>=0,

$$J(\theta+\eta d)\ge J(\theta),$$

strictly if eta>0 and at least one d_k>0 (finite logits ensure all q_k are interior).

**Proof.** On a reached stage M_k=X_k and its score is X_k-q_k. If X_k=1, its centered binary milestone advantage is nonnegative; the protected terminal bonus is either positive or zero. Thus total credit is nonnegative, as is the score. If X_k=0, Y=0, both usable stage credit and terminal bonus are nonpositive, as is the score. Their product is nonnegative in either case. Placement uses Y=X_3 and has the same property. Nonnegative weighting, averaging and zero masks preserve this sign. Therefore every logit can only increase under raw ascent, each q_k can only increase, and their product cannot decrease.

**What the result explains.** In a setting with separable stage decisions, group normalization and balanced positive weights do not destroy the improvement mechanism. Setting kappa=0 removes the direct negative terminal-credit pressure on already-successful prefixes. This is a precise reason for the design, not an assurance that earlier skills can never deteriorate.

This is a sufficient result, not a necessity result: it does not establish that symmetric credit fails in this same separable model, or that protection is superior there. The real-system comparison must establish that benefit.

**What breaks the guarantee.** In the real robot, a stage has many actions: a successful prefix can contain poor actions and its scores have no fixed sign. Parameters are shared across stages/scenes, so gradients interfere. Candidate selection changes state distributions. Adam, entropy pressure, mean-action penalties, and finite off-policy updates change the direction. Inference uses another exploration law. None of these are covered by the three-gate proposition.

**Symmetric-credit counterexample at the row level.** Suppose every candidate reaches a milestone and only one of eight completes the task. Its stage advantage is zero. A failed-completion candidate's terminal advantage is -1/sqrt(7). With kappa=1 this gives negative credit even to the achieved earlier gate; kappa=0 makes that bonus zero. This proves removal of one negative term, not improved whole-task performance in general.

### Exact bridge from protection to the shared-network update

At the behavior parameters (all ratios equal one), on one fixed retained batch with frozen weights and advantages, let d_kappa be the raw policy-surrogate ascent gradient. Let R contain approach/pickup rows whose own milestone was achieved and whose usable full-task advantage is negative. Then

$$d_\kappa-d_1=\lambda(1-\kappa)\sum_{r\in R}w_r(-A_{r3})\,\psi_r,\qquad\psi_r=\nabla\log\pi_\theta(u_r\mid h_r,b_r).$$

**Proof.** Only protected negative bonus terms differ: kappa A-A=(1-kappa)(-A). At ratio one, the local derivative of the unclipped/clipped surrogate is the advantage times the policy score. Sum the differences with the fixed row weights.

Consequently,

$$\|d_\kappa-d_1\|\le\lambda(1-\kappa)\sum_{r\in R}w_r(-A_{r3})\|\psi_r\|,\quad 0\le\kappa\le1.$$

This identifies exactly which gradient component the preservation rule removes. Its inner product with the full-task gradient is not necessarily positive for a shared neural network. Measure that inner product and stage-gradient interference rather than infer retention from positive scalar credit. Off-policy clipping, a changed retained batch, or optimizer preconditioning require recomputing the actual direction. This identity is a direct algebraic consequence of the implemented rule, not an automatic novelty claim.

## 5. Proposition D: the actual condition for neural-policy improvement

For a fixed success objective J with L-Lipschitz gradient in a neighborhood, any realized update Delta whose connecting segment lies in that neighborhood satisfies

$$J(\theta+\Delta)-J(\theta)\ge\nabla J(\theta)^\top\Delta-\frac L2\|\Delta\|^2.$$

**Proof.** Integrate grad J along the segment and bound the remainder using gradient Lipschitzness. No assumption about how Delta was computed is needed.

For Delta=eta d, if a=grad J dot d>0, then 0<eta<2a/(L||d||^2) gives improvement. This includes a direction resulting from the *actual optimizer*, not just a symbolic loss gradient.

A useful error form is d=c grad J+e with c>0. If ||e||<=rho||grad J|| and rho<c, then

$$\nabla J^\top d\ge(c-\rho)\|\nabla J\|^2,\qquad \|d\|\le(c+\rho)\|\nabla J\|,$$

so an improving range is

$$0<\eta<\frac{2(c-\rho)}{L(c+\rho)^2}.$$

The implementation changes listed above enter e. The current reports do not bound rho or L; substituting assumed small values would create a false certificate. Neural contact dynamics need not make sampled returns differentiable pathwise, but score differentiation and local smoothness of *expected* success still require stated regularity.

For a random direction with mean dbar, the expected lower bound is eta grad J dot dbar - (L eta^2/2)(||dbar||^2+tr Cov(d)), under the same smoothness/integrability assumptions. Correct expectation alone does not control finite-sample update noise.

### Independent directional audit

Freeze a proposed d, then collect independent on-policy audit groups for the same objective. With full trajectory scores and a leave-one-out terminal baseline,

$$\widehat a=\frac1G\sum_i(Y_i-b_{-i})\,d^\top\sum_t\nabla\log\pi_\theta(u_{it}|h_{it},b_{it}),\quad b_{-i}=\frac1{G-1}\sum_{j\ne i}Y_j,$$

has expectation grad J dot d. The baseline is independent of trajectory i conditional on c. Do not estimate this on the batch that selected d, or treat repeated scenes as independent. A confidence interval requires an explicit statistical method and appropriate moments/bounds; it is an empirical alignment diagnostic, not an automatic finite-sample guarantee for unbounded Gaussian scores.

The method's theorem-to-experiment link should be: measure alignment and update scale, then test strict success on a separate evaluation panel. Paired finite-difference evaluation along d is an alternative when full-score gradients are too expensive. Separately test deterministic-mean deployment. A sampled KL average or PPO clipping does not establish the above smoothness/alignment condition; compare the precise role of bounds in [TRPO](https://arxiv.org/abs/1502.05477).

## 6. Supporting results and explicit limitations

### Composition requires preserving prefixes

For nested necessary stage events E1, E2, Y,

$$J=P(E1)P(E2\mid E1)P(Y\mid E2),\qquad \nabla\log J=\sum_{k=1}^3\nabla\log p_k,$$

when all factors are positive. This is a chain-rule identity, with no independence assumption. Improving completion conditional on lift can be outweighed by losing pickup probability. For example, 0.8 x 0.8 x 0.5=0.32 exceeds 0.8 x 0.5 x 0.7=0.28. Conditional success can also change because a different subset of scenes survives. Use paired scene analysis.

Equal weighting of binary milestone rewards is not generally equivalent to maximizing J: always reaching/lifting and placing 10% has milestone reward 2.1, whereas completing all stages on 50% and otherwise none has reward 1.5. The latter has higher strict success. The full-task bonus reduces a specific omission of terminal credit but does not remove every objective mismatch.

### Self-imitation has a conditional justification

Fix a scene and successful-trajectory distribution q_beta^+(tau)=P_beta(tau)Y/J_beta, J_beta>0. Under common dynamics and well-defined likelihood ratios,

$$\log\frac{J_\theta}{J_\beta}\ge\mathbb E_{q_\beta^+}\log\frac{P_\theta(\tau)}{P_\beta(\tau)}.$$

**Proof.** Restrict the new policy's successful mass to the old successful support to obtain J_theta/J_beta >= E_q[P_theta/P_beta], then apply Jensen. A nonnegative population expected log-likelihood gain is sufficient for per-scene nondecrease. Finite success-only banks, smoothed actions on old observations, MSE, mixed/scripted teachers and shared scene compromises do not automatically satisfy it. Keep this as supporting context unless SFT is a main contribution.

### Residual reachability explains representational limits

For fixed scalar p and s_r>0,

$$\overline{\{\mu_\theta\}}\subseteq[\tanh(p-s_r),\tanh(p+s_r)],$$

with equality only if the residual network can reach the full scalar logit range. A desired deterministic mean outside that interval cannot be fit by this bounded head, regardless of data. Gaussian exploration still has full latent support. Report saturation and compare the VLA prior's benefits against this constraint; this does not prove the failed unbounded-correction arm should perform better.

### Hardware transfer is a separate premise

For aligned simulated/real trajectory laws and the same measurable outcome rubric,

$$|J_{\rm real}(\pi)-J_{\rm sim}(\pi)|\le\operatorname{TV}(P_{\rm real}^\pi,P_{\rm sim}^\pi).$$

**Proof.** Success is an event, and TV bounds differences of event probabilities. This is meaningful only if the mismatch is independently bounded; unknown contact/perception mismatch makes it uninformative. An outcome-label mismatch adds its own discrepancy. Simulator learning theory cannot certify physical success in advance.

## 7. What belongs in the letter

- Mathematical formulation and **Proposition A**, with a short trajectory-factorization proof.
- **Proposition B**, as a signal-availability lemma, attributed alongside binary-GRPO theory.
- **Proposition C**, as a declared sequential-gate benchmark of the actual protected credit rule, with the shared-network limitation explicit.
- **Proposition D**, as the conditional bridge to real policy updates, supported by an independent alignment/step-size experiment rather than unverified constants.
- Compact stage success factorization to motivate retention measurements.

Keep lengthy derivations in this internal note while deciding what fits the six-to-eight-page article. It is not an extra manuscript appendix that bypasses RA-L's page limit. None of these standard identities is automatically a novel theorem. The potentially distinctive contribution is the analysis and testing of this particular combination of protected stage credit, candidate-balanced sampling, persistent exploration, and robotic manipulation. A broader novelty review remains necessary.

## 8. Proof-to-code and proof-to-evidence map

Source paths below are relative to `../../RL_VLA_Bootstrapping/` from this note.

| Mathematical component | Code | What must be verified experimentally |
|---|---|---|
| Conditional latent score | `rl_vla_bootstrapping/policy/latent_correction_policy.py`, `latent_gaussian_log_prob` | Frozen prior, stored unclipped samples/offset gates, full versus selected scores |
| Binary stage groups | `rl_vla_bootstrapping/policy/rank_local_grpo.py`, `torch_group_advantages`; `mjwarp_rank_local_collector.py`, `three_stage_group_credit` | Independent candidates, epsilon/clip settings, unusable and non-finite exclusions |
| Achieved-stage protection | `mjwarp_rank_local_collector.py`, `add_full_task_bonus` | Resolved lambda/kappa; symmetric/protected matched comparison |
| Stage/candidate weighting | `rank_local_grpo.py`, `global_stage_loss_weights` | Retained candidates/row counts; no post-selection recentering |
| Implemented surrogate | `smolvla_grpo_finetune_cdpr.py`, stage-weighted MJWarp optimizer loop | Regularizers, Adam/preconditioning, clipping, minibatch staleness |
| Improvement condition | Not enforced by current code | Independent directional alignment, uncertainty, actual update displacement, strict evaluation |
| Physical transfer | Not guaranteed by training code | Actual hardware trials under the declared protocol |

Run `python3 theory/verify_mathematics.py` from the submission folder for independent finite-model sanity checks. These check algebra and counterexamples; they do not train or certify the CDPR policy. See `theory/verification.json` for actual results.
