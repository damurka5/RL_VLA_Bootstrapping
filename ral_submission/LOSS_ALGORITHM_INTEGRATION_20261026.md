# Loss and algorithm integration - 26 October 2026

The active Method section now contains a compact actor objective: asymmetric PPO/GRPO clipping with epsilon_minus=0.20 and epsilon_plus=0.28, plus Gaussian entropy regularization. The text explains its stabilizing role without claiming an isolated ablation result. The reaching-only coefficient is zero; pickup, placement and composition use 0.0002. The unused action-L2 penalty is no longer listed as part of the reported update.

Algorithm 1 summarizes dense skill acquisition, binary refinement, successful-experience banking, scheduled balanced self-imitation and milestone-based full-task learning. It is included from algorithms/staged_pipeline_paper.tex. Stage budgets and stopping rules remain explicit training-manifest inputs, consistent with the existing missing historical protocol information. Numerical automatic WX200 thresholds are not attributed to the CDPR lineage.

The more detailed WX200 driver and CDPR update algorithms remain separate supporting snippets. Training code and numerical results are unchanged.
