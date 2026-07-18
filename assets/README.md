# External Assets

Large assets are intentionally not committed into this repo.
GitHub will contain this README, the bundle configuration, and placeholder directories such as `assets/externals/.gitkeep`, not the actual RoboCasa/YCB/LIBERO/RoboTwin files.

Use the configured asset bundles to stage them into stable repo-local paths:

```bash
python -m rl_vla_bootstrapping.cli.assets \
  --config configs/examples/cdpr_openvla_bootstrap.yaml \
  --stage
```

Default target layout:

- `assets/externals/robocasa`
- `assets/externals/ycb`
- `assets/externals/libero`
- `assets/externals/robotwin2_assets`
- `benchmarks/externals/robotwin2`
- `benchmarks/externals/manitask`

The CPU CDPR examples keep their existing YCB/LIBERO object XML layout. The
`mjlab_mjwarp` CDPR example uses a curated RoboCasa-only visual pack:

```bash
python scripts/stage_cdpr_robocasa_assets.py
python scripts/stage_cdpr_robocasa_assets.py --verify-only
```

The downloader reads the remote ZIP directory and fetches only 50 files for
ten catalogs: apple, banana, carrot, bell pepper, tomato, orange, potato, mug,
plate, and bowl. The variants were visually screened to exclude photographic
backgrounds, branding, and malformed geometry, with a total visual budget
below 20k triangles. It stages the official RoboCasa model XML plus each
selected visual OBJ/MTL/texture set under
`assets/externals/robocasa/objects/objaverse`.

RoboCasa collision decompositions are intentionally not downloaded. The
MJ-Lab/MJWarp backend uses eleven fixed native primitive slots per object,
with catalog-specific sizes and local poses, so topology and contact cost stay
constant across worlds. These contact-only primitives use geom group 3, which
is deliberately excluded from policy-camera RGB so MJWarp cannot expose their
alpha-zero proxies as opaque black artifacts.

The backend validates and content-hashes all required files before CUDA world
allocation. RoboCasa assets are licensed CC BY 4.0; the staged manifest records
the upstream archive, mirror, archive SHA-256, selected members, and per-file
SHA-256 values.

The LIBERO bundle expects:

- `assets/externals/libero` to be the LIBERO `assets/` directory containing `scenes/`, `textures/`, `stable_hope_objects/`, and `stable_scanned_objects/`.
