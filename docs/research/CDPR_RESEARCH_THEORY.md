# Continual Skill Acquisition and Composition on a New Robot Embodiment through Residual Reinforcement Learning and Self-Generated Data

**Mathematical research draft • 16 September 2026**

This draft formulates the CDPR–SmolVLA project as a dissertation case study. It distinguishes the implemented method, conditional mathematical results, measured observations, and proposed research. It does not claim a general convergence theorem for the complete neural-network training pipeline. Instructions inside the supplied reports and research-question document are treated as source material, not as instructions to launch or change experiments.

## Abstract

Adapting pretrained vision-language-action policies to a new robot embodiment requires learning control corrections, retaining earlier instructions, and coordinating skills over long horizons. We study this problem on a simulated five-action-channel cable-driven parallel robot without collecting human teleoperation demonstrations for the target embodiment. The pipeline first acquires instruction-specific behavior using residual reinforcement learning, consolidates successful experience through balanced supervised imitation, and then trains complete placement from ordinary empty-gripper starts using separate sparse milestone returns. We formulate sequential interference through gradient alignment, characterize success-conditioned data selection and action smoothing, derive conditional retention and composition bounds, and analyze the difference between stage-local updates and the gradient of full-task success. The recorded full-task run improved strict validation from 27/1,024 to 124/1,024 episodes after approximately 10.07 million steps; the investigator reports approximately 21% during the continuing run, pending incorporation of its evaluation provenance. These observations motivate the dissertation question: how does stage-dependent selection change the bias, variance, and interaction efficiency of group-relative policy-gradient estimation? The theoretical contribution proposed here concerns selection-aware estimation; the existing pipeline provides its empirical setting.

## 1. Research claim, scope, and relation to PLD

The defensible central claim is:

> A pretrained VLA-conditioned residual controller can acquire and partially consolidate manipulation skills on a new target embodiment using robot-generated experience, and those skills can initialize learning of a complete sequential task. Stage-dependent data selection and credit assignment govern which acquired behavior is preserved and which downstream behavior receives a learning signal.

“Without human demonstrations” means **no newly collected human teleoperation dataset for the target embodiment in the reported adaptation pipeline**. It does not mean the pretrained VLA has never seen human-generated robot data. It also does not mean the system has no human-designed rewards, simulator state, action mappings, calibration, or controllers. Historical banks include scripted-oracle actions and alignment controllers; distinguish policy self-imitation from autonomous scripted supervision in every data manifest. To claim that a particular result is exclusively policy-generated, establish that property for its complete checkpoint and dataset lineage.

The supplied paper is Xiao et al., *Self-Improving Vision-Language-Action Models with Data Generation via Residual RL*, published at ICLR 2026 after a 2025 preprint. Its PLD framework uses residual specialists, base-policy probing, autonomous collection, and VLA distillation. Its §4.5 presents generalization mechanisms as hypotheses supported by experiments, rather than proving every empirical improvement. Its real-world Franka setup starts with 200 teleoperated trajectories (§4.4); the later autonomous phase must be distinguished from that initialization. [PLD project and publication record](https://wenlixiao.com/self-improve-VLA-PLD), [paper](https://arxiv.org/abs/2511.00091).

| Dimension | PLD | This project |
|---|---|---|
| Learning correction | Off-policy residual actor–critic | Primarily GRPO-style residual policy updates in the relevant campaign |
| Data generation | Base-policy prefixes followed by residual recovery | Successful policy rollouts; historical scripted-oracle banks; continuous multi-teacher chains with recorded bridges |
| Consolidation | SFT into the VLA generalist | Residual SFT; optional action-expert LoRA SFT; recent full-task RL freezes LoRA |
| Main empirical question | Quality and distribution of generated data | Acquisition, forgetting, retention, and complete-task composition on CDPR |
| Proposed theory focus | Reference methodology | Selection bias and variance in stage-dependent group updates |

The pipeline resemblance is prior art. Residual RL, replay, behavior cloning, skill sequencing, and inverse-probability weighting are not individually new contributions. A prospective contribution is a well-specified selection mechanism with an estimator analysis and matched-budget evidence. This draft does not establish an exhaustive novelty claim.

## 2. Mathematical setting and the actual policy

### 2.1 Embodiment-specific partially observed control

Let the target embodiment be $e$ and define

$$
\mathcal M_e=(\mathcal S,\mathcal A_e,P_e,\mathcal O,O_e,\rho_0,H).
\tag{1}
$$

Here $s_t$ contains physical and controller state, $o_t$ is the available observation, $h_t=(o_0,a_0,\ldots,o_t)$ is the observation history, and $\ell$ is the instruction. Include persistent task predicates, action queues, and controller memory in the state when needed for the Markov property. The policy need only use its implemented features $x_t=f(h_t,\ell)$; writing a full-history policy does not assert that the current residual network is recurrent. Any privileged simulator features supplied to the actor must be disclosed as part of $x_t$.

For CDPR, the normalized command is

$$
a_t=(a_t^x,a_t^y,a_t^z,a_t^{\mathrm{yaw}},a_t^{\mathrm{grip}})\in[-1,1]^5.
\tag{2}
$$

A calibrated mapping $u_t=C_e(s_t,a_t)$ turns this into physical controller input; $P_e$ includes the resulting dynamics. Changing embodiment changes $C_e$, $P_e$, proprioception, cameras, and feasible actions. This is adaptation to one new embodiment, not yet a demonstration of broad transfer across many embodiments.

Let $c=(s_0,\ell,\text{scene parameters})\sim\mathcal D_{\mathrm{eval}}$. Fix this distribution, the horizon, and the outcome definition for comparisons. The principal objective is

$$
J(\theta,\phi)=\mathbb E_{c,\tau\sim p_{\theta,\phi}(\cdot\mid c)}[Y(\tau)]
=\Pr(Y=1),\qquad Y\in\{0,1\}.
\tag{3}
$$

The strict full-task outcome follows the shared implementation:

$$
Y=N\land G\land L\land U\land\neg S\land\neg W,
\tag{4}
$$

where $N$ is native placement, $G$ a recorded physical grasp, $L$ a held lift, $U$ a valid release, $S$ a carry-slip latch, and $W$ a wrong-placement latch. These are temporally updated events, not six arbitrary final-state predicates. Native geometry and release timing come from the simulator contract. Reach diagnostics are separate from strict success.

### 2.2 Residual parameterization and action chunks

Let $p_\phi(h_t,\ell,\xi_t)\in\mathbb R^{K\times5}$ be the mapped SmolVLA prior chunk, including its noise draw $\xi_t$. The actual deterministic residual mean is

$$
r_\theta(x_t,p_t)=\tanh f_\theta([x_t,\operatorname{vec}(p_t)]),\qquad
\mu_{\theta,\phi}(x_t,p_t)=\tanh(p_t+s_r r_\theta(x_t,p_t)).
\tag{5}
$$

The outer and inner $\tanh$ matter: this is not simply an unconstrained additive controller. The reviewed SFT path emits eight slots and executes four before replanning; unused and post-terminal slots need masks. In trajectory likelihoods, index sampled chunks or their conditionally sampled slots consistently. Do not score the same random decision again for every repeated physics substep.

The recent full-task RL leg optimizes $\theta$ and exploration parameters while keeping $\phi$ fixed, including loaded action-expert LoRA. The SFT path can update $\phi$ through LoRA,

$$
W=W_0+\frac{\alpha}{r}BA,
\tag{6}
$$

but this optional stage must be distinguished from residual-only SFT and residual-only RL.

**Proposition 1 — residual mean reachability.** For fixed scalar prior $p$ and scale $s_r>0$, every representable deterministic mean lies inside

$$
\mathcal I(p)=[\tanh(p-s_r),\tanh(p+s_r)].
\tag{7}
$$

Consequently, action-space squared error on target $a^*$ is bounded below by $\operatorname{dist}(a^*,\mathcal I(p))^2$.

**Proof.** The inner $\tanh$ lies in $(-1,1)$ and the outer $\tanh$ is monotone. The displayed closed interval is the closure of the reachable scalar means. Minimizing squared distance over that interval gives the lower bound. Summing over supervised coordinates gives the vector bound. Shared network parameters can create additional approximation error. $\square$

This is a restriction on the mean and SFT fit, not on the support of Gaussian exploration. It explains why more residual SFT epochs cannot fit every target and why changing the prior through LoRA can change representability.

### 2.3 A likelihood qualification needed for honest gradient proofs

The reviewed sampler draws a Gaussian command and clips it to $[-1,1]$, then evaluates a Gaussian density at the clipped command. This is a surrogate likelihood: a clipped Gaussian has point masses at its boundaries. In one dimension,

$$
\Pr(a=-1)=\Phi_N\!\left(\frac{-1-\mu}{\sigma}\right),\quad
\Pr(a=1)=1-\Phi_N\!\left(\frac{1-\mu}{\sigma}\right),
\tag{8}
$$

and the interior uses the Gaussian density. $\Phi_N$ denotes the standard normal CDF. A density evaluated at $a=\pm1$ is not the corresponding boundary mass.

The unbiased-gradient results below assume an exact score for the sampling distribution. They do not certify the present clipping approximation. Exact alternatives include retaining and scoring the pre-clipping latent sample, using the mixed clipped distribution, or a consistently implemented squashed Gaussian. If a common random exploration offset persists across an episode, its joint/conditional likelihood also needs treatment; multiplying independent marginal action densities generally omits temporal dependence. No training code is changed by this document.

## 3. Step 1 — acquiring basic instructions separately and sequentially

Let $\ell_k$ denote a skill instruction, for example reaching, pickup, or placement from a held-object start. Its training distribution $\rho_{k,m}$ can change with curriculum rung $m$. Define

$$
J^{\mathrm{train}}_{k,m}(\theta)=
\mathbb E_{s_0\sim\rho_{k,m},\tau\sim\pi_\theta}[R_{k,m}(\tau)],\qquad
\theta_{k,0}=\theta_{k-1,\mathrm{end}}.
\tag{9}
$$

Historical dense rewards fit this expression; do not retroactively describe all early runs as sparse-binary training. Skill success is separately $J_k^{\mathrm{eval}}=\Pr_{\rho_k^{\mathrm{eval}}}(Y_k=1)$. A held-object placement reward learns transport/release under a different initial distribution from the complete instruction.

For a regular differentiable stochastic policy with environment dynamics independent of $\theta$,

$$
\nabla J_k=\mathbb E\!\left[R_k(\tau)\sum_{t=0}^{H-1}
\nabla_\theta\log\pi_\theta(a_t\mid h_t,\ell_k)\right].
\tag{10}
$$

**Derivation.** Differentiate $\int p_\theta(\tau)R_k(\tau)d\tau$ under the integral, use $\nabla p=p\nabla\log p$, and factor the trajectory density into environment and policy terms. Environment terms have zero parameter derivative. Terminated trajectories can be padded by parameter-independent absorbing transitions. $\square$

**Proposition 2 — a local learning guarantee, with assumptions.** If $J_k$ has an $L_k$-Lipschitz gradient and an update uses the exact gradient, then

$$
J_k(\theta+\eta\nabla J_k)\ge J_k(\theta)
+\eta\left(1-\frac{L_k\eta}{2}\right)\|\nabla J_k\|^2.
\tag{11}
$$

**Proof.** Apply the smooth-function lower bound $J(\theta+d)\ge J(\theta)+\nabla J^\top d-L\|d\|^2/2$ with $d=\eta\nabla J$. Thus $0<\eta<2/L_k$ gives improvement unless the gradient is zero. For an unbiased noisy estimate $\widehat g$, the expected lower bound contains the additional penalty $L_k\eta^2\operatorname{tr}\operatorname{Cov}(\widehat g)/2$. $\square$

This proves why a sufficiently small accurate update can improve the current objective. It does not prove global convergence of GRPO, mastery of a new embodiment, or that a curriculum is always more efficient than direct training. Those are empirical comparisons. Familiar priors may improve exploration, but a restricted residual can also prevent useful corrections.

For group size $G$, the implemented family of updates starts from

$$
\overline R=G^{-1}\sum_iR_i,\quad
\widehat A_i=\frac{R_i-\overline R}{\sqrt{G^{-1}\sum_j(R_j-\overline R)^2}+\epsilon},
\tag{12}
$$

and maximizes a clipped importance-ratio surrogate, with separately configured entropy and action penalties. Random normalization and clipping make this different from (10). GRPO originates in [DeepSeekMath](https://arxiv.org/abs/2402.03300); its name does not confer an unbiasedness or monotonic-improvement guarantee.

## 4. Step 2 — losing previous instructions during new-skill training

Record a retention matrix $A_{k,i}=J_i^{\mathrm{eval}}(\theta_k)$, using the same evaluation protocol for skill $i$ across checkpoints. Define observed forgetting as

$$
F_{k,i}=\max_{j<k}A_{j,i}-A_{k,i}.
\tag{13}
$$

Negative values indicate an improvement over previous measurements. Sampling noise and changes in resets or scoring must be separated from parameter forgetting.

**Proposition 3 — gradient interference.** Under an update $\theta'=\theta+\eta g_k$, and an $L_i$-smooth old objective,

$$
J_i(\theta')-J_i(\theta)
=\eta\nabla J_i(\theta)^\top g_k+\varepsilon_i,
\quad |\varepsilon_i|\le\frac{L_i\eta^2}{2}\|g_k\|^2.
\tag{14}
$$

If $\nabla J_i^\top g_k=-a<0$, then the old objective strictly decreases whenever $0<\eta<2a/(L_i\|g_k\|^2)$.

**Proof.** Taylor's theorem with the Lipschitz-gradient remainder gives (14). Its upper bound is $-\eta a+L_i\eta^2\|g_k\|^2/2<0$ under the stated condition. $\square$

Sequential updates share parameters, so instructions can request conflicting changes. Reaching may tolerate closed fingers, while a pickup-ready approach requires an open gripper. A policy trained to carry an already held object need not preserve the earlier approach. This is a mechanism consistent with the results, not proof that measured gradient conflict caused each historical decline. Estimate gradient inner products at matched checkpoints to test that causal explanation. Gradient constraints and memory replay are established continual-learning ideas; see [Gradient Episodic Memory](https://proceedings.neurips.cc/paper/2017/hash/f87522788a2be2d171666752f97ddebb-Abstract.html).

For example, the report records move-to success of 0.080 before Cycle 1 consolidation and 0.311 afterward. The dedicated 0.6299 reaching result uses its own registered evaluation protocol; it should not automatically be treated as the numerator of a matched forgetting comparison.

## 5. Step 3 — collecting self-imitation data and optionally smoothing it

### 5.1 Success-conditioned collection is a change of distribution

For a source policy $\beta$ and a fixed scene/instruction $c$, let $p_\beta(\tau\mid c)$ denote the rollout distribution and $J_\beta(c)>0$ its success probability. The ideal successful-trajectory distribution is

$$
q_\beta^+(\tau\mid c)=
\frac{p_\beta(\tau\mid c)Y(\tau)}{J_\beta(c)}.
\tag{15}
$$

An empirical bank approximates this distribution using successful rollouts, together with source-policy IDs and task-specific labels. In this project, “SIL” primarily means behavior cloning of selected robot-generated experience. It is not automatically the positive-advantage actor–critic algorithm of [Oh et al., Self-Imitation Learning](https://proceedings.mlr.press/v80/oh18b.html).

**Proposition 4 — success filtering favors easy scenes.** If candidates are generated from $c\sim\mathcal D$, the scene distribution among retained successes is

$$
q(c\mid Y=1)=\frac{\mathcal D(c)J_\beta(c)}{
\mathbb E_{c'\sim\mathcal D}J_\beta(c')}.
\tag{16}
$$

**Proof.** Bayes' rule. $\square$

Thus a large bank of successes can contain poor coverage of difficult scenes or bowls. Rebalancing instruction counts changes the intended imitation objective, but does not invent missing successful transitions. Successful prefix banks must carry prefix/stage labels; their existence is not evidence of full-task completion.

### 5.2 Why self-imitation can improve behavior in an idealized case

**Proposition 5 — a conditional population self-imitation improvement result.** Fix $c$, a behavior policy $\beta$ with $J_\beta(c)>0$, the same environment, and exact unsmoothed successful trajectories. If a candidate policy satisfies

$$
\mathbb E_{q_\beta^+}\!\left[\log\frac{p_\theta(\tau\mid c)}{p_\beta(\tau\mid c)}\right]\ge0,
\tag{17}
$$

with well-defined likelihood ratios, then $J_\theta(c)\ge J_\beta(c)$.

**Proof.** Integrating over successful trajectories in the support of $p_\beta$ gives

$$
\frac{J_\theta(c)}{J_\beta(c)}
\ge\mathbb E_{q_\beta^+}\frac{p_\theta}{p_\beta}.
$$

Jensen's inequality then gives

$$
\log\frac{J_\theta(c)}{J_\beta(c)}
\ge\log\mathbb E_{q_\beta^+}\frac{p_\theta}{p_\beta}
\ge\mathbb E_{q_\beta^+}\log\frac{p_\theta}{p_\beta}\ge0.
$$

Environment likelihoods cancel in the ratio, leaving the sum of policy action log-likelihood differences. $\square$

Exact population maximum-likelihood fitting can satisfy (17) if the behavior policy is feasible and optimization improves on it. Finite samples, shared-parameter compromises, smoothing, MSE in place of the correct likelihood, scripted-teacher mixtures, and missing support all prevent importing the guarantee directly. An average log-improvement over scenes bounds a geometric-mean success ratio; it does not by itself prove improvement in arithmetic-mean success. This is a precise reason to distinguish an ideal self-improvement mechanism from the measured pipeline.

### 5.3 What the smoothing operation actually does

For selected continuous command channels and odd width $w=2m+1$, the moving average is

$$
\widetilde a_t=\frac1w\sum_{j=-m}^m a_{\operatorname{clip}(t+j,0,T-1)}.
\tag{18}
$$

The implementation filters each live episode along its stored environment-action axis with edge padding; the gripper is excluded by default. The reports sometimes call this a five-decision filter, but the implementation's array axis is authoritative. Specify whether a “step” is an action sample, decision, or physics substep in experiments. It is an offline noncausal filter. Near boundaries or discontinuities, “zero phase” does not imply unchanged event timing.

**Proposition 6 — denoising helps only when its bias is small enough.** Suppose one channel has $a_t=u_t+\epsilon_t$, independent zero-mean noise with variance $\sigma^2$, and an interior window with distinct samples. Then

$$
\mathbb E(\widetilde a_t-u_t)^2
=\left(\frac1w\sum_{j=-m}^{m}u_{t+j}-u_t\right)^2+\frac{\sigma^2}{w}.
\tag{19}
$$

Therefore smoothing improves squared error relative to the unsmoothed sample exactly when the squared bias is less than $\sigma^2(1-1/w)$.

**Proof.** Expand the squared error into deterministic bias and averaged noise. The cross term has zero expectation; independent noise variances sum to $w\sigma^2/w^2$. $\square$

For correlated noise replace $\sigma^2/w$ by $w^{-2}\sum_{i,j}\operatorname{Cov}(\epsilon_{t+i},\epsilon_{t+j})$. Edge padding duplicates samples, so the independent-distinct-sample expression does not apply there. With scalar $u$ having $|u''|\le M$ and spacing $\Delta t$, a symmetric interior window has bias at most $M\Delta t^2m(m+1)/6$. At grasp and release transitions the smooth-signal model may be false. Lower action jitter is not automatically lower Cartesian jerk, cable tension variation, or contact failure.

**Proposition 7 — a local trajectory robustness condition.** Suppose, within one unchanged contact/controller regime,

$$
\|F(s,a)-F(s',a')\|\le L_s\|s-s'\|+L_a\|a-a'\|.
$$

For identical starts and $\|\widetilde a_t-a_t\|\le\delta$,

$$
\|\widetilde s_t-s_t\|\le L_a\delta\sum_{j=0}^{t-1}L_s^j.
\tag{20}
$$

**Proof.** The error obeys $e_{t+1}\le L_se_t+L_a\delta$, with $e_0=0$; induction gives (20). If each required predicate has a positive robustness margin exceeding its Lipschitz constant times the state-error bound, its truth value is preserved. $\square$

Contact switches and latch changes can invalidate the regime assumption; use replay to test actual survival. The data pipeline should replay modified commands, retain successful replay trajectories, and pair each new action with its new pre-action observation. Attaching smoothed commands to old frames generally creates inconsistent supervision. Refresh the current student's prompt-conditioned priors on the replayed frames; an inference refresh alone is not physics replay.

Measured evidence is limited but useful: the Phase 3 report gives moving-average w5 replay survival of 0.909 for pickup and 0.938 for placement, alongside command-delta reductions of 0.582 and 0.596. It performs better than the tested EMA and median alternatives on those summaries. The unsmoothed survival control is 1.0, so smoothing still loses successes. These numbers establish a smoothness–survival tradeoff, not a causal improvement in downstream SFT. The later three-stage bank specification starts with unsmoothed executed actions.

## 6. Step 4 — balanced SFT and recovery of earlier skills

Let $\mathcal B_k$ denote a retained bank for instruction $k$. Let $z$ index stage, $d$ destination, $b$ action coordinate, and $m_{tjb}$ denote a valid target-coordinate mask. Use a row loss

$$
\ell_{\mathrm{BC}}(\theta,\phi;x,p,A)=
\frac{\sum_{j,b}m_{tjb}(\mu_{\theta,\phi}(x,p)_{jb}-A_{jb})^2}
{\sum_{j,b}m_{tjb}}.
\tag{21}
$$

Rows with no valid targets are excluded. Mask unused chunk slots, terminal tails, and invalid stage-boundary targets. Current-prior refresh makes $p$ consistent with the student's parameters and instruction; a source teacher's prior is not interchangeable with the student's prior.

For retention weights $\omega_k$ summing to one,

$$
\mathcal L_{\mathrm{ret}}=\sum_k\omega_k\mathbb E_{\mathcal B_k}\ell_{\mathrm{BC}},
\quad
\mathcal L_{\mathrm{SFT}}=\lambda\mathcal L_{\mathrm{ret}}
+(1-\lambda)\sum_{z,d}\omega_{zd}\mathbb E_{\mathcal B_{zd}}\ell_{\mathrm{BC}}.
\tag{22}
$$

The second term is for a new stage/destination bank and is optional in earlier retention cycles; its nonnegative weights $\omega_{zd}$ sum to one over represented strata, and $\lambda\in[0,1]$. A uniform row sampler assigns implicit task weight $n_k/\sum_jn_j$, where $n_k$ is the number of rows. Long reaching trajectories can dominate even when episode counts are balanced. Deliberate weights define which skills SFT is asked to preserve; they do not guarantee each skill's success improves.

**Proposition 8 — a conditional retention bound.** Compare a teacher $\beta_k$ and student $\pi$ in the same task, with horizon $H$. Let $d^{\beta_k}_t$ be the teacher's history distribution. Then

$$
|J_k(\pi)-J_k(\beta_k)|
\le\min\left\{1,\sum_{t=0}^{H-1}
\mathbb E_{h\sim d^{\beta_k}_t}
\operatorname{TV}(\pi(\cdot\mid h),\beta_k(\cdot\mid h))\right\}.
\tag{23}
$$

If average teacher-occupancy KL, in the direction $\beta_k\Vert\pi$, is at most $\varepsilon_k$, then

$$
|J_k(\pi)-J_k(\beta_k)|\le\min\{1,H\sqrt{\varepsilon_k/2}\}.
\tag{24}
$$

**Proof.** Couple teacher and student actions maximally at identical histories, and use the same environment randomness until their first different action. A union bound over first mismatches gives (23); binary outcomes can differ only after such a mismatch. Pinsker's inequality bounds each TV by the square root of half its KL; Jensen's inequality over time and teacher histories gives (24). $\square$

This connects imitation fidelity to retained task success under explicit distributional assumptions. It is a finite-horizon coupling argument, not an application of a measured training MSE as if it were a success certificate. Distribution shift in imitation learning is a central issue in [Ross, Gordon and Bagnell](https://proceedings.mlr.press/v15/ross11a.html).

For unbounded Gaussians with the same fixed covariance $\sigma^2I$, $D_{\mathrm{KL}}=\|\mu_\beta-\mu_\pi\|^2/(2\sigma^2)$, relating MSE to this bound. Deterministic controllers, unequal covariance, and clipping require different treatment. If bank occupancy $q_k$ covers teacher occupancy with $d^{\beta_k}_t(h)\le C_kq_{k,t}(h)$, then bank-average KL $\varepsilon_{B,k}$ implies the corresponding bound with $C_k\varepsilon_{B,k}$. A successes-only bank may have no such finite coverage constant.

Recovery has a clear interpretation: SFT restores behavior recorded before it was overwritten. It need not discover new successful states. A nonconvex shared controller can still trade one skill against another. The recorded Cycle 1 changes are move-to 0.080→0.311, plate 0.500→0.523, and bowl 0.2305→0.2765 under its registered family protocols. Later SFT also reduced composed-task success, and expanded Arm C's lower imitation loss coincided with only 5/128 strict completions. Preserve these negative results; they delimit the mechanism.

## 7. Step 5 — composing skills and training with staged rewards

### 7.1 Physical composition is a distributional interface problem

Represent a conceptual skill as an option $o_k=(\mathcal I_k,\pi_k,\beta_k^{\mathrm{term}})$, with initiation set, policy, and termination condition. This describes teacher sequencing; it does not assert that the final student has separately parameterized options. A shared student can use one full-task prompt throughout and realize phases implicitly.

For approach, pickup, and placement let $E_k$ be completion of stage $k$ in order within the remaining common horizon. Then

$$
\Pr(Y=1)=p_1p_2p_3,
\quad p_1=\Pr(E_1),\quad
p_2=\Pr(E_2\mid E_1),\quad
p_3=\Pr(Y=1\mid E_1,E_2),
\tag{25}
$$

provided the selected stage events are necessary for $Y$. Zero-probability conditioning is handled by observing that full success is then zero.

**Proof.** Apply the probability chain rule to nested successful-prefix events. No independence assumption is used. $\square$

Standalone pickup rates are measured under a reset distribution $\rho_2$; the composed policy starts pickup under the approach policy's handoff distribution $\nu_2$. These are different. If the next-stage success function $V_k(s)$ lies in $[0,1]$, then

$$
|\mathbb E_{\nu_k}V_k-\mathbb E_{\rho_k}V_k|
\le\operatorname{TV}(\nu_k,\rho_k).
\tag{26}
$$

**Proof.** This is the variational characterization of TV for functions bounded between zero and one. $\square$

Thus, if standalone success is $q_k$ and handoff mismatch is at most $\delta_k$, conditional composed success is at least $\max(0,q_k-\delta_k)$, provided remaining time, controller memory, instructions, and all other state variables are included consistently. Multiplying these lower bounds yields a conditional composition bound. A reaching tolerance of 2 cm does not imply a pickup-ready tolerance of roughly 1.3 cm; the local report documents exactly this mismatch.

Teacher chains must therefore maintain one continuous physical state, controller state, contact history, and remaining budget. Concatenating unrelated arrays or calling every successful pickup a successful placement does not create a valid full-task demonstration. Relabeling to the final prompt is justified only for goal-consistent transitions, and partial traces remain partial.

### 7.2 The implemented milestone objective

Let monotone trajectory returns be $M_1,M_2,M_3\in\{0,1\}$, with $M_3\le M_2\le M_1$:

1. Reach the object under the registered threshold or establish persistent grasp.
2. Establish a current held lift, using the 5 cm lift contract.
3. Complete strict full placement under (4).

These describe the repaired milestone logic. An optional open-hand approach conjunction is diagnostic and is not a requirement of strict success. A finite-state progress variable $z_t$ makes history-dependent milestones Markov in the augmented state.

A useful *analytical comparison*, not the exact implementation, is the scalar auxiliary objective

$$
J_{\mathrm{aux}}(\theta)=\mathbb E\left[\sum_{k=1}^3w_kM_k\right]
=\sum_kw_k\Pr(M_k=1).
\tag{27}
$$

The current implementation computes a separate group advantage $\widehat A_{ik}$ from each $M_{ik}$, assigns action records to a phase, filters usable groups, and balances represented stages and candidates. Its fixed-batch policy surrogate can be written

$$
\mathcal S_{\mathrm{stage}}(\theta)=
\frac1{|\mathcal K_B|}\sum_{k\in\mathcal K_B}\frac1{n_k}
\sum_{i\in\mathcal C_k}\frac1{T_{ik}}
\sum_{t\in\mathcal T_{ik}}
\min\{\rho_{it}(\theta)\widehat A_{ik},
\operatorname{clip}(\rho_{it}(\theta),1-\epsilon_-,1+\epsilon_+)\widehat A_{ik}\}.
\tag{28}
$$

Here $\mathcal K_B$ is the represented-stage set, $\mathcal C_k$ the retained candidates with stage-$k$ rows, $n_k=|\mathcal C_k|$, $T_{ik}=|\mathcal T_{ik}|$, and $\rho_{it}$ the stored-behavior/new-policy likelihood ratio. The implemented optimizer additionally uses entropy and mean-action penalties. Global distributed weights implement the candidate/stage mean; their unbiased minibatch sampling concerns this fixed empirical surrogate, not necessarily $\nabla J$.

Stage-restricted credit omits downstream reward effects on earlier actions; candidate-length averaging and stage normalization also alter the weighting. Therefore (28) is not generally an unbiased gradient estimator of either (3) or (27), even with perfect action likelihoods. This distinction is central to the dissertation.

### 7.3 Why milestones can supply signal earlier

**Proposition 9 — probability of a nonconstant binary group.** For conditionally independent rollouts in a fixed scene with success probability $p$,

$$
P_{\mathrm{mixed}}(p,G)=1-p^G-(1-p)^G.
\tag{29}
$$

**Proof.** The only constant binary groups are all successes and all failures, with probabilities $p^G$ and $(1-p)^G$. $\square$

For $G=8$, $p=0.03$ gives approximately 21.63% mixed groups, while $p=0.30$ gives approximately 94.23%. Easier milestones can increase informative groups when their probabilities move toward the interior. A saturated milestone at $p\approx1$ has little contrast. Across heterogeneous scenes use $\mathbb E_c[P_{\mathrm{mixed}}(p(c),G)]$, not $P_{\mathrm{mixed}}(\mathbb E_cp(c),G)$. Mixed groups still need relevant action rows, accurate likelihoods, and a non-erased advantage to yield useful gradients.

### 7.4 What staged rewards cannot guarantee

**Counterexample — more progress reward can mean less full success.** Policy A always reaches and lifts but places successfully with probability 0.1. Policy B succeeds in all three stages with probability 0.5 and otherwise reaches none. For equal milestone weights,

$$
J_{\mathrm{aux}}(A)=2.1>1.5=J_{\mathrm{aux}}(B),
\qquad J(A)=0.1<0.5=J(B).
\tag{30}
$$

Rewarding intermediate progress can prefer the worse full-task policy. Even when every individual successful trajectory receives more reward than every failure, expected-policy rankings need not agree. For the implemented stage surrogate, an update direction $g_s$ is locally beneficial for full success only when $\nabla J^\top g_s>0$ with a small enough step, by the same smoothness argument as (14). This is an alignment condition to measure, not an established property of current training.

**Proposition 10 — objective-preserving shaping alternative.** For an augmented state $\bar s_t$, time-indexed potential $\Phi_t$, and matching discount factor,

$$
r'_t=r_t+\gamma\Phi_{t+1}(\bar s_{t+1})-\Phi_t(\bar s_t),
\qquad \Phi_H=0,
\tag{31}
$$

gives

$$
\sum_{t=0}^{H-1}\gamma^tr'_t
=\sum_{t=0}^{H-1}\gamma^tr_t-\Phi_0(\bar s_0).
\tag{32}
$$

**Proof.** The potential terms telescope. With a fixed initial distribution, the remaining term is policy-independent. Use zero terminal potential on early termination or pad to the horizon consistently. $\square$

For the undiscounted success objective use $\gamma=1$. This is a finite-horizon form of established [potential-based reward shaping](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf), offered as a comparison arm. The current stage-local GRPO loss is not this construction. Also, with purely episodic REINFORCE and a constant starting potential, total-return shaping can cancel in the baseline and supply no extra group contrast; policy invariance alone does not ensure a learning advantage.

## 8. The dissertation connection — selection-aware group estimation

Keep the PhD question focused:

> How does stage-dependent rollout or action-record selection change the bias–variance tradeoff of group-relative policy-gradient estimators, and under what computational constraints can selection-aware estimation improve interaction efficiency for sparse-reward VLA-conditioned control?

The case study contains two related but distinct selection problems: success filtering changes the imitation-data distribution (§5), while stage/group/action filtering changes the RL update distribution. Inverse-probability weighting of RL records does not undo missing imitation-data coverage.

### 8.1 An exact reference estimator

At one fixed sampling policy, set $\psi_{it}=\nabla\log\pi_\theta(a_{it}\mid h_{it},c)$ and $Y_i$ to strict full-task success. For $G>1$ conditionally independent rollouts of the same scene,

$$
b_{-i}=\frac1{G-1}\sum_{j\ne i}Y_j,\qquad
\widehat g_{\mathrm{LOO}}=\frac1G\sum_i\sum_t(Y_i-b_{-i})\psi_{it}.
\tag{33}
$$

**Proposition 11 — unbiased leave-one-out reference.** Under exact on-policy scores and ordinary differentiation regularity, $\mathbb E\widehat g_{\mathrm{LOO}}=\nabla J$.

**Proof.** Conditional on $c$ and other rollouts, $b_{-i}$ is independent of rollout $i$. The expected score of a normalized trajectory distribution is zero, so $\mathbb E[b_{-i}\sum_t\psi_{it}\mid c]=0$. The reward term is (10). Average over $i$ and $c$. $\square$

This is the reference motivated by [RLOO](https://aclanthology.org/2024.acl-long.662/). Independence is conditional on the shared scene; sharing adaptive random variables between candidates needs additional analysis. Including the rollout's own reward in an unnormalized mean baseline instead gives the exact scaling $(G-1)/G$ of the expected reward gradient. Dividing by the random group standard deviation introduces another, generally nonconstant, change. These are separate effects from filtering.

### 8.2 The observed erasure mechanism

For $G=8$ and returns $(1,0,0,0,0,0,0,0)$, population-standard-deviation normalization gives

$$
\widehat A_1=\sqrt7,\qquad
\widehat A_{2:8}=-1/\sqrt7.
\tag{34}
$$

If only the successful candidate contributes downstream rows, every retained advantage equals $\sqrt7$. Subtracting their row mean makes them all zero. The calculation is exact when the normalizer has no epsilon; any common positive value is still erased by centering. Removing that second centering repairs erasure but does not make the surviving records an unbiased full-task gradient. This mechanism is recorded in the September 13 audit and repaired in the later run.

### 8.3 Selection correction and what it recovers

Let the complete collected batch be $B$, define

$$
v_{it}=(Y_i-b_{-i})\psi_{it},
$$

and retain record $(i,t)$ with indicator $M_{it}$ and known probability $q_{it}=\Pr(M_{it}=1\mid B)>0$. Then

$$
\widehat g_{\mathrm{SC}}=\frac1G\sum_{i,t}\frac{M_{it}}{q_{it}}v_{it}.
\tag{35}
$$

**Proposition 12 — selection correction.** $\mathbb E[\widehat g_{\mathrm{SC}}\mid B]=\widehat g_{\mathrm{LOO}}(B)$, hence $\mathbb E\widehat g_{\mathrm{SC}}=\nabla J$ under Proposition 11.

**Proof.** Conditional on $B$, every $v_{it}$ and $q_{it}$ is fixed, and $\mathbb E[M_{it}/q_{it}\mid B]=1$. Apply linearity and then total expectation. Selection indicators need not be independent for this mean result. $\square$

Without weighting but retaining the original denominator, conditional bias is $G^{-1}\sum_{i,t}(q_{it}-1)v_{it}$. Dividing instead by the random number of retained rows adds ratio-estimator bias. Post-selection baseline recomputation changes the $v_{it}$ themselves and falls outside (35). Deterministic removal has $q=0$; weighting cannot recover a nonzero omitted contribution. Deterministically discarded zero-contribution records are harmless for that reference, but group rules require checking rather than assuming this exception.

Equation (35) corrects record selection from a complete collected batch. It does not recover unvisited downstream states, make teacher actions on-policy, fix an incorrect likelihood, reverse action-length averaging, or convert milestone rewards into strict full-task reward. All rollouts, including discarded groups and assisted prefixes, still cost simulator interactions.

### 8.4 The variance cost and a constrained allocation result

**Proposition 13 — thinning adds variance.** If record indicators are conditionally independent Bernoulli draws, then

$$
\operatorname{Cov}(\widehat g_{\mathrm{SC}}\mid B)
=\frac1{G^2}\sum_{i,t}\left(\frac1{q_{it}}-1\right)v_{it}v_{it}^{\top}.
\tag{36}
$$

**Proof.** $\operatorname{Var}(M/q)=1/q-1$; cross-covariances vanish under the stated conditional independence. Apply covariance linearity. $\square$

By total covariance, the unconditional covariance equals that of the complete-batch estimator plus the expectation of (36). Thinning an already available unbiased batch therefore cannot lower its statistical variance. Correlated or fixed-size selection needs pairwise inclusion probabilities in the covariance calculation.

Suppose including a record has processing cost $c_{it}>0$ and impose $\sum q_{it}c_{it}\le C$ with a feasible positive floor $q_{\min}$. Minimizing the conditional variance trace from (36) yields, for interior probabilities,

$$
q_{it}^{*}=\operatorname{clip}_{[q_{\min},1]}
\left(\frac{\|v_{it}\|}{\sqrt{\lambda c_{it}}}\right),
\tag{37}
$$

where $\lambda$ enforces the budget when it binds.

**Derivation.** Discard constants and minimize $\sum\|v_{it}\|^2/q_{it}+\lambda\sum c_{it}q_{it}$. Differentiation gives $-\|v_{it}\|^2/q_{it}^2+\lambda c_{it}=0$; convexity and the box constraints give clipping. Zero-contribution records can use the floor without loss. $\square$

This is a classical importance-sampling allocation applied to the reference estimator, not a claim of a new theorem. Computing every $\|v_{it}\|$ may itself consume the processing budget. A research method could estimate stage-conditional second moments using a small pilot or earlier batches, choose positive stage probabilities, and correct inclusion weights. Proxy allocation preserves conditional unbiasedness when the actual probabilities are known, but loses the exact variance optimality claim.

The testable hypothesis is improvement over **uncorrected filtering at a matched relevant budget**, not lower variance than processing all records for free. Statistical MSE and learning efficiency are separate endpoints:

$$
\operatorname{MSE}(\widehat g)=\|\mathbb E\widehat g-\nabla J\|^2+
\operatorname{tr}\operatorname{Cov}(\widehat g),\quad
N_\alpha=\inf\{N:J(\theta_N)\ge\alpha\}.
\tag{38}
$$

Count $N$ as all training simulator action steps, including failed/discarded candidates, collection and replay used for training, and assisted prefixes. Report optimizer cost and wall time separately. Offline SFT has no new environment actions, but its source data were not free. Runs that do not reach $\alpha$ are censored, not silently removed.

The nearby preprint [Prism-GRPO](https://arxiv.org/abs/2608.17423), submitted 18 August 2026, changes reward using execution-quality scores to split same-outcome groups and studies alignment. It is relevant competition, but reward augmentation differs from inclusion-probability correction under a fixed binary outcome. It should be compared where its quality-signal assumptions can be matched; its existence prevents presenting “recovering zero-advantage groups” alone as novelty.

## 9. A unified algorithm description

**Inputs:** pretrained VLA and embodiment mapping; registered skill/full-task predicates; disjoint collection, teacher-selection, validation, and test scenes; instruction list; interaction and optimization budgets.

1. **Acquire.** Sequentially optimize a shared residual policy for each basic instruction under its declared curriculum. Save skill checkpoints and evaluate every previously acquired instruction after each leg.
2. **Collect.** Roll out retained teachers, label source-policy versus scripted-controller actions, and preserve successful skills and physically continuous stage transitions. Keep full-task and successful-prefix banks distinct.
3. **Optionally smooth.** Filter selected command channels within an episode; replay; retain valid successes; use replayed pre-action observations and commands. Keep an unsmoothed control and report rejected trajectories and collection costs.
4. **Consolidate.** Refresh current-student inputs under the correct prompts, split by scene, and fit masked actions with declared instruction/stage/destination weights. Train residual and optional LoRA stages explicitly. Select by rollout success and retention.
5. **Compose.** Start the student with an empty gripper away from the object and one final instruction. Optimize the declared three-stage surrogate; compute strict success separately. Evaluate on the same start distribution without teacher handoff or servo assistance.
6. **Study selection.** As a separate dissertation experiment, hold the strict reward and start distribution fixed; compare exact-likelihood unfiltered RLOO, conventional GRPO, uncorrected selection, and selection correction. Do not attribute reward, likelihood, termination, or reset repairs to the new estimator.

Steps 1–5 summarize the research pipeline, with historical variants disclosed. Step 6 is proposed work. Automatic monotonic improvement after every complete cycle is not claimed.

## 10. Evidence ledger and experiments that would support the claims

### 10.1 What is established by the supplied record

| Observation | Supported statement | Limit |
|---|---|---|
| Dedicated reaching: 645/1,024, 0.6299 | A target-embodiment skill was learned with the reported residual pipeline | Specific scenes, tolerance, and curriculum; not broad embodiment generalization |
| Cycle 1: move-to 0.080→0.311; plate 0.500→0.523; bowl 0.2305→0.2765 | Balanced SFT recovered measured family performance | Caught placement and family-specific protocols; not complete placement |
| w5 survival: pickup 0.909; placement 0.938 | Smoothing often preserves recorded success while reducing command variation | Does not establish better trained-policy success |
| Expanded Arm C: 5/128 strict, 20 lifts | More downstream data/lower fitting loss did not clear its continuation gate | Distinct 128-chain protocol |
| Full-task staged GRPO: 27→124/1,024 after 10.07M steps | Same-run strict validation improves from 2.6% to 12.1% | In-run validation, last-five mean 9.6%, shared scenes; no untouched final-test result |
| Investigator's current report: approximately 21% | Interim progress reported on 16 September | Checkpoint, denominator, seeds, plate/bowl split and scoring provenance not supplied with this request |

Do not compare historical 70–80% caught-placement scores to current complete-task scores. Earlier reset protocols sometimes started inside the placement radius and omitted approach. The 2026-09-14 ledger is more specific than older “composed” labels. Also, the old-termination latch is a diagnostic on current trajectories, not necessarily a counterfactual rerun with the old termination rule.

### 10.2 Minimum experimental structure

| Question | Comparison | Primary measurements |
|---|---|---|
| Does sequential acquisition help? | Direct full-task RL vs curriculum-initialized RL; identical total accounting | Full success versus all interactions, per-stage entry rates |
| Is forgetting parameter interference? | Before/after checkpoints on identical tasks; optional replay/projection control | Retention matrix, paired changes, gradient alignment |
| Does smoothing help SFT? | Same raw source episodes: identity vs w5; replay both; common-survivor and complete-pipeline comparisons | Jitter, replay survival, retained scene coverage, trained-policy success |
| Does balanced replay recover skills? | No replay vs natural-frequency vs instruction-balanced SFT | Full retention vector and full-task success; same source bank and optimizer budget |
| Which trainable component matters? | Residual-only vs residual plus LoRA | Same scenes and targets, reachable error floor, held-out rollouts |
| Do stages help beyond repairs? | Strict terminal reward vs scalar milestones vs phase-local credit | Same initial checkpoint, resets, horizon, termination and likelihood |
| Does correction improve estimation? | Exact unfiltered RLOO vs uncorrected thinning vs corrected thinning | Exact-gradient bias/MSE in a small sequential model; compute costs |
| Does correction improve learning? | Correct GRPO, RLOO, uncorrected/corrected selection, compatible recent baseline | Success curves, $N_\alpha$, all interactions and wall time |

For smoothing, the common-survivor comparison isolates the change in targets on matched episodes; the complete-pipeline comparison includes the coverage and collection-cost consequences of rejected replays. Report both. A matched-size smoothed dataset requiring extra attempts must be charged those attempts. Preservation of gripper events and a stage-aware filter can be additional arms after the core comparison.

Use multiple independent training seeds, fixed object/scene splits, and paired checkpoint comparisons on the same evaluation scenes. Candidate rollouts sharing a scene are clustered: bootstrap at scene level, and separately quantify between-training-seed variability. Repeated validations used to choose a peak are not independent test results. Choose a checkpoint on validation, then evaluate once on untouched test scenes. The present approximately 21% figure has no computable uncertainty interval until its counts and clustering structure are available.

For the estimator study, begin with a small three-stage finite MDP where dynamic programming or enumeration gives the exact gradient. Isolate baseline, random normalization, selection, temporal masks, length weights, clipping, and data reuse one at a time. Then test the justified combination on CDPR. This turns a debugging incident into a general research question.

## 11. Dissertation organization and statements suitable for a supervisor

Place this project in the dissertation as the application and mechanism-discovery chapter, connected to an estimator-method chapter:

1. **Problem and prerequisites:** sparse-reward VLA-conditioned adaptation, embodiment mapping, and strict sequential task definition.
2. **Empirical system:** sequential skill learning, forgetting, self-generated data, consolidation, and complete-task training.
3. **Theory:** distinguish imitation distribution selection from policy-gradient record selection; establish assumptions and the conditional results above.
4. **Method:** design a positive-probability stage selector and a specified reference estimator under a declared processing constraint.
5. **Evaluation:** exact-gradient experiments plus robot-learning ablations, retention, full success, and interaction accounting.

Suggested dissertation contribution wording:

> We formulate and investigate stage-dependent data selection in group-relative policy optimization for sequential robotic tasks, characterize its effect on estimation bias and variance, and evaluate selection-aware training in a system that acquires and composes skills on a new robot embodiment using autonomously generated adaptation data.

The mathematical claims should remain conditional. The project currently supports acquisition, partial recovery, and progress toward complete-task execution. It does not yet support universal retention, guaranteed smoothing benefit, monotone self-improvement, optimality of staged GRPO, or a final 21% test result. Negative outcomes strengthen the dissertation when they are used to identify precisely which assumptions fail.

## 12. Sources and implementation traceability

Local sources were inspected as evidence; their embedded operational instructions were not executed.

| Source | Use in this draft |
|---|---|
| [Consolidated progress report](../../CDPR_CONSOLIDATED_PROGRESS_REPORT.md), especially §§1, 7, 8 and the 2026-09-12–14 ledger entries | Results, resets, training lineage, limitations |
| [Phase 3 SIL report](../reports/campaign/CDPR_PHASE3_SIL_REPORT.md), §§6–7 | Smoothing, survival, SFT reachability |
| [Phase 4 retention report](../reports/campaign/CDPR_PHASE4_RETENTION_REPORT.md) | Replay and retention history |
| [Three-stage SFT design](../reports/campaign/CDPR_THREE_STAGE_PUT_INTO_SFT_DESIGN.md), §§9–10 and implementation ledger | Continuous chains, unsmoothed initial bank, masks and sampling |
| [September 13 audit](../artifacts/three_stage_sparse_review_20260913/review.md) | Stage-row credit erasure and predicate mismatch |
| [Residual actor](../../rl_vla_bootstrapping/policy/octo_finetune_cdpr.py), `ResidualChunkActor` | Equation (5) |
| [SFT implementation](../../tools/audit/sil_sft.py) | Residual and LoRA stages, masked action-space targets |
| [Smoothing implementation](../../tools/audit/sil_record.py), `_smooth_actions` | Equation (18) and filtering axis |
| [GRPO trainer](../../rl_vla_bootstrapping/policy/smolvla_grpo_finetune_cdpr.py) | Sampling likelihood qualification and update surrogate |
| [Collector](../../rl_vla_bootstrapping/policy/mjwarp_rank_local_collector.py), `three_stage_group_credit` | Separate milestone returns and filtering |
| [Distributed stage weighting](../../rl_vla_bootstrapping/policy/rank_local_grpo.py), `global_stage_loss_weights` | Stage/candidate/row normalization |
| [Full-task outcome](../../rl_vla_bootstrapping/simulation/cdpr_full_task_outcome.py) | Equation (4) |
| [RQ.md](</Users/damirnurtdinov/Downloads/RQ.md>) | PhD scope, reference estimator, interaction-cost definition |
| [Attached PLD paper](</Users/damirnurtdinov/Downloads/22285_Self_Improving_Vision_La.pdf>), §§2–3, 4.4–4.5 and Algorithm 1 | Reference structure and methodological differences |

External sources are linked beside the claims they support. The named mathematical propositions in this draft are elementary derivations or specializations of established results, unless explicitly identified as a proposed empirical hypothesis. They should not be presented as newly discovered theorems. The draft follows the supplied project record rather than independently reconstructing every remote experiment.

**Draft verification.** Equation numbering and local source links were checked. Exact enumeration in a Bernoulli group model verified the RLOO expectation, own-mean baseline scaling, and inverse-probability correction. Numerical checks reproduced singleton-stage credit erasure, mixed-group probabilities, and the finite-horizon potential telescoping identity. These checks establish consistency of those calculations; they do not verify that the robot experiments satisfy every theorem assumption.
