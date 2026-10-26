# Method-centered manuscript revision — 26 October 2026

The six top-level sections are Introduction, Related Work, Methods, Experiments, Conclusion, and Future Work.

Problem motivation and task definitions now open Methods in prose. CDPR-specific action semantics and success criteria are in Experiments. CDPR setup and results, skill retention, LIBERO setup and results, and the ongoing WX200 study are grouped under Experiments.

Four displayed expressions remain: balanced action-regression loss, the 40d prior, clipped RL loss minus entropy, and mixed-outcome group probability. The actor parameterization, detailed credit equations, reward-weight equation, trajectory objective, and Bernoulli-gate argument are removed from the active manuscript. Milestone composition is explained in words and a simplified editable diagram.

Algorithm 1 keeps all dense skill stages before all binary stages, followed by full-task composition. Stage targets, plateaus, and budget exhaustion are described without numerical configuration values. Automatic stopping and consolidation acceptance are distinguished from the historical manual CDPR sequence.

The paper no longer mentions CSV files, source-report provenance, training manifests, or checkpoint comparisons. Evaluation sample sizes, update budgets, terminal reward weights, and limitations needed to interpret results remain. Evidence files and implementation audits are unchanged. WX200 contains image slots and a location for quantitative results; no rates are invented.

The native diagrams.net sources, combined XML, vector exports, and ZIP are synchronized. The generator retains the simplified labels and reader-facing Table II caption.
