#!/usr/bin/env python3
"""Why does a composed container episode not start on the desk?

`extract_cdpr_transition_demonstrations` refuses an episode whose object does
not begin within a centimetre of `support_surface_z + target_rest_height`,
because a clip whose lift is measured from a pose the object was never resting
at is not a pick-up demonstration. On the `bowl_peak` harvest that rejected 341
of 1632 container episodes -- 21%, the second largest block after the honest
`no_pickup_success`, and large enough that it is worth knowing whether those
episodes are unusable or the datum is wrong.

WHAT THE ANSWER LOOKS LIKE, AND WHY THE SPLIT IS BY OBJECT

The composed reset places the target at `support_surface_z + rest_height[:, 0]`
and the predicate's datum is `rest_height[world, target_slot]`; both read the
same per-catalog table, so a mismatch that depends on WHICH object was spawned
points at that table rather than at the reset. A uniform offset across every
catalog points the other way -- at the reset height, or at the object settling
between the reset and the first recorded observation.

That last one is why `dz step0->1` is here. `object_xyz[0]` is captured one env
step after the reset, so an object spawned at a height its mesh does not
actually rest at has already begun to fall by the time anything is recorded. A
large negative `dz` beside a positive `z0-desk` is an object dropped from too
high, which is a reset bug and also perturbs the scene the policy sees.

CPU only, on recordings already on disk.

Usage::

    python tools/audit/desk_start_probe.py 'runs/bowl_peak_harvest_*/record_*.npz'
"""

from __future__ import annotations

import collections
import glob
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
os.environ.setdefault("MUJOCO_GL", "disable")

import numpy as np  # noqa: E402

from tools.audit.sil_record import _catalog_name, _instruction_name  # noqa: E402

CONTAINER_IDS = (3, 4)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    pattern = argv[0] if argv else "runs/bowl_peak_harvest_*/record_*.npz"
    tolerance = float(argv[1]) if len(argv) > 1 else 0.01

    paths = sorted(glob.glob(pattern))
    if not paths:
        raise SystemExit(f"No recordings matched {pattern!r}.")

    rows: dict[tuple[str, str], list] = collections.defaultdict(list)
    for path in paths:
        data = np.load(path)
        ids, slots = data["instruction_ids"], data["target_slots"]
        worlds = np.arange(len(slots))
        z0 = data["object_xyz"][0, worlds, slots, 2].astype(np.float64)
        z1 = data["object_xyz"][1, worlds, slots, 2].astype(np.float64)
        desk = (
            data["support_surface_z"] + data["target_rest_height"]
        ).astype(np.float64)
        rest = data["target_rest_height"].astype(np.float64)
        # Absent on recordings written before the field existed; reported as
        # `unknown` rather than guessed, since the per-object split is the
        # whole question.
        has_catalog = "target_catalog_ids" in data.files
        catalogs = (
            data["target_catalog_ids"] if has_catalog
            else np.full(len(slots), -1)
        )
        for world in np.flatnonzero(np.isin(ids, CONTAINER_IDS)):
            key = (
                _instruction_name(int(ids[world])),
                _catalog_name(int(catalogs[world])) if has_catalog else "unknown",
            )
            rows[key].append(
                (z0[world] - desk[world], z1[world] - z0[world], rest[world])
            )

    header = (
        f"{'instruction / object':38} {'n':>5} {'z0-desk p50':>12} {'p10':>10} "
        f"{'p90':>10} {'>tol':>6} {'dz step0->1 p50':>16} {'rest_h':>8}"
    )
    print(f"[desk] {len(paths)} recordings, tolerance {tolerance} m", flush=True)
    print(header)
    print("-" * len(header))
    total = over_total = 0
    for key in sorted(rows):
        block = np.array(rows[key])
        pct = lambda column, p: float(np.percentile(block[:, column], p))  # noqa: E731
        over = int((np.abs(block[:, 0]) > tolerance).sum())
        total += len(block)
        over_total += over
        print(
            f"{key[0] + ' / ' + key[1]:38} {len(block):5d} {pct(0, 50):12.5f} "
            f"{pct(0, 10):10.5f} {pct(0, 90):10.5f} {over:6d} "
            f"{pct(1, 50):16.5f} {block[0, 2]:8.4f}"
        )
    print("-" * len(header))
    print(
        f"container episodes {total}, outside the tolerance {over_total} "
        f"({over_total / max(total, 1):.4f})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
