# Response to author notes - 22 October 2026

1. **Main idea:** the title and narrative now focus on adapting VLA-based control to new embodiments. The method first explains dense-reward initial skill learning, binary refinement and self-imitation, then full-task staged RL. Low-level likelihood and credit details follow as implementation choices. Skill instructions and full-task instructions are explicitly separated.
2. **Simple abstract:** removed negative-completion-bonus protection, conditional latent likelihood terminology, intermediate comparisons, action counts, exact denominators and confidence intervals. The abstract states the training idea and current 61.43% endpoint, identified as simulation development evaluation. Hardware remains visibly assumption-based.
3. **LIBERO numbers:** 65.3%, 61.0%, and 61.7% come directly from Table I on page 2 of the supplied `main (6).pdf`. They compare runs within that supplied study, not outside papers. The source describes LIBERO-Spatial goal verification with OpenVLA-OFT. These numbers are not for CDPR reaching or its complete instruction. The source's dense phase-aware reward is separate from the main pipeline's dense-to-binary transition. Numbers remain in the dedicated LIBERO results and are absent from the abstract/conclusion.
4. **Outcome rather than training history in the abstract:** the current selected SmolVLA controller's 61.43% is retained. 20M/40M training counters and intermediate checkpoint comparisons remain only in experimental detail for reproducibility. The method is about VLA adaptation, with the current robot as one test embodiment.
5. **Old Figure 1:** removed. OpenVLA-OFT was not run on the CDPR and did not share its scenes. The revised experiment introduction says this explicitly. A future overview, if wanted, should depict the learning sequence itself, not two supposedly matched robot branches.
6. **Role of the cable robot:** introduced briefly as an unconventional embodiment whose gripper is positioned through coordinated cable actuation. Its role is to test adaptation under a substantial mechanical/control change. No unsupported application story or claim of demonstrated universal robot transfer has been added.

## Precise implementation point still to confirm

Which weights were reinitialized for SmolVLA -> CDPR: the complete action expert, action projections, or the new residual/controller head? The inspected final controller uses a residual MLP and retained VLA components. This is not interchangeable with resetting the full pretrained action expert. The general problem and proposed learning schedule are written now; the exact intervention remains visibly pending.

## Evidence priorities added

The data checklist now prioritizes per-skill dense/binary switch records, self-imitation/retention tables, action-module initialization, and a staged-vs-terminal full-task composition comparison. Existing likelihood and correction-network diagnostics are secondary to these main method claims.
