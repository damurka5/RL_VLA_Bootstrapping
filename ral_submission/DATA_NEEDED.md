# Missing evidence — updated 28 October 2026

The supplied October 19–23 TensorBoard log now supports the 60M in-run
endpoint (723/1024 = 70.61%) and panel peak (728/1024 = 71.09%). Separate
evaluation of this endpoint, its resolved run manifest, checkpoint bytes/hash,
scene IDs and per-episode outcomes remain unavailable. Retain the 40M control's
61.43% as the latest separate development result. This log closes no
independent-seed, held-out evaluation or real-world evidence gap below.

1. Training manifest: skill rewards, resets, dense-to-binary switches, SIL ordering, data origins and budgets. Archive the resolved configuration and control timing.
2. Exact LIBERO phase predicates, progress functions, aggregation and filtering code. Use the new CSV weights, w_p = w_c = 0.05.
3. Independent seeds and evaluation records. Aggregate LIBERO endpoints have 300 rollouts; task_1 initialization has 1,000, but final denominators remain missing.
4. Matched retention evaluations and balanced-SIL controls. Figure 5 uses rounded report rates and separates forgetting from recovery.
5. WX200 action/state mapping, actual apparatus or CAD, camera views, five-DoF annotation and timestamped task frames.
6. Physical trial logs including failures, interventions, resets and deployment/adaptation condition.

Do not add intermediate-model comparisons to the main text. New charts should answer questions about retention, composition bottlenecks or matched sample efficiency.
