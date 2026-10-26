# Response to author notes - 22 October 2026

1. **Main idea:** the title and narrative now focus on adapting VLA-based control to new embodiments. The method first explains dense-reward initial skill learning, binary refinement and self-imitation, then full-task staged RL. Low-level likelihood and credit details follow as implementation choices. Skill instructions and full-task instructions are explicitly separated.
2. **Simple abstract:** removed negative-completion-bonus protection, conditional latent likelihood terminology, intermediate comparisons, action counts, exact denominators and confidence intervals. The abstract states the training idea and current 61.43% endpoint, identified as simulation development evaluation. Hardware remains visibly assumption-based.
3. **LIBERO numbers:** 65.3%, 61.0%, and 61.7% come directly from Table I on page 2 of the supplied `main (6).pdf`. They compare runs within that supplied study, not outside papers. The source describes LIBERO-Spatial goal verification with OpenVLA-OFT. These numbers are not for CDPR reaching or its complete instruction. The source's dense phase-aware reward is separate from the main pipeline's dense-to-binary transition. Numbers remain in the dedicated LIBERO results and are absent from the abstract/conclusion.
4. **Outcome rather than training history in the abstract:** the current selected SmolVLA controller's 61.43% is retained. 20M/40M training counters and intermediate checkpoint comparisons remain only in experimental detail for reproducibility. The method is about VLA adaptation, with the current robot as one test embodiment.
5. **Old Figure 1:** removed. OpenVLA-OFT was not run on the CDPR and did not share its scenes. The revised experiment introduction says this explicitly. A future overview, if wanted, should depict the learning sequence itself, not two supposedly matched robot branches.
6. **Role of the cable robot:** introduced briefly as an unconventional embodiment whose gripper is positioned through coordinated cable actuation. Its role is to test adaptation under a substantial mechanical/control change. No unsupported application story or claim of demonstrated universal robot transfer has been added.

## Action-module initialization resolved from code and reports

Following the author’s instruction to inspect code and reports, the audit establishes that the SmolVLA action expert and projections are loaded from pretrained weights. A separate residual MLP is initialized from scratch at initial skill learning; later phases warm-start its weights and inherited LoRA. The selected continuation freezes the VLA/LoRA and trains the bounded residual and exploration parameters. The old OpenVLA continuous-head experiments and the unselected zero-initialized correction candidate are separate branches. See `evidence/ACTION_MODULE_AUDIT_20261022.md`.

The abstract now says learning a new action module; the method gives the actual initialization and trainable components. Resetting a whole pretrained expert remains an untested extension. This preserves the dense -> binary/self-imitation -> staged-RL idea while matching the evaluated implementation. Exact historical run manifests remain reproducibility inputs, not an unresolved architectural question.

## Evidence priorities added

The data checklist now prioritizes per-skill dense/binary switch records, self-imitation/retention tables, action-module initialization, and a staged-vs-terminal full-task composition comparison. Existing likelihood and correction-network diagnostics are secondary to these main method claims.
