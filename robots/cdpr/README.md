# CDPR Example

This folder contains the full example embodiment bundle for the 5-DoF cable-driven parallel robot:

- `cdpr_mujoco/`: robot XML, controller, and scene switcher.
- `cdpr_dataset/`: reward functions, RL environment, and synthetic scene/task helpers.

The framework treats this as just one robot bundle under `robots/`. New robots should follow the same pattern instead of placing robot-specific files at the repository root.

Generated scene wrappers should be treated as bundles: the main wrapper XML may reference companion helper XMLs in the same directory using the same filename stem prefix.
