# External Assets

Large assets are intentionally not committed into this repo.
GitHub will contain this README, the bundle configuration, and placeholder directories such as `assets/externals/.gitkeep`, not the actual YCB/LIBERO/RoboTwin files.

Use the configured asset bundles to stage them into stable repo-local paths:

```bash
python -m rl_vla_bootstrapping.cli.assets \
  --config configs/examples/cdpr_openvla_bootstrap.yaml \
  --stage
```

Default target layout:

- `assets/externals/ycb`
- `assets/externals/libero`
- `assets/externals/robotwin2_assets`
- `benchmarks/externals/robotwin2`
- `benchmarks/externals/manitask`

The CDPR example expects:

- `assets/externals/ycb` to contain the `ycb/*.xml` objects or the `ycb` directory directly.
- `assets/externals/libero` to be the LIBERO `assets/` directory containing `scenes/`, `textures/`, `stable_hope_objects/`, and `stable_scanned_objects/`.
