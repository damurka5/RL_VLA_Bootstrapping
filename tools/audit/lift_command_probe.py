#!/usr/bin/env python3
"""Did the commanded lift move, and did the achieved lift follow it?

`pick_up` succeeds by lifting the object 5 cm while holding it, and §4.2 puts
the loaded plant's ineffective->reliable transition at a commanded a_z of
0.20-0.30. So the family's success rate is downstream of one number: the MEAN
a_z the policy holds over the steps the object is actually gripped. Measured on
bowl_peak, that number is +0.232 for `pick_up` and +0.371 / +0.272 for plate
and bowl -- pick_up's median sits inside the transition band while both
containers clear it, which is the ordering of their lift rates.

This reports that number per instruction, per arm, beside the grasp rate, the
lift rate conditioned on grasping, and the peak lift achieved. Point it at a
before/after pair and it says whether an intervention moved the COMMAND, which
is a different question from whether it moved the success rate and the only one
that distinguishes "the mechanism worked and the budget was short" from "the
mechanism did not engage".

Lift is measured from `object_xyz[0]`, not from `initial_target_positions`,
which is stale for uncaught container starts.

CPU only, on recordings already on disk.

Usage::

    python tools/audit/lift_command_probe.py \
        'runs/release_recovery_pilot_*/baseline/record_*.npz' \
        'runs/release_recovery_pilot_*/final_eval/record_*.npz'
"""

from __future__ import annotations

import os
import glob
import sys

import numpy as np

_ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
os.environ.setdefault("MUJOCO_GL", "disable")
NAMES = {3: "bowl", 4: "plate", 8: "pick_up"}

for arm in sys.argv[1:]:
    acc = {k: [] for k in NAMES}
    tot = {k: [0, 0, 0] for k in NAMES}
    files = sorted(glob.glob(arm))
    if not files:
        print(f"{arm}: no recordings")
        continue
    for path in files:
        d = np.load(path)
        ids, slots = d["instruction_ids"], d["target_slots"]
        worlds = np.arange(len(slots))
        held = d["caught_target"].astype(bool) & d["active"].astype(bool)
        tz = d["object_xyz"][:, worlds, slots, 2]
        lift = tz - tz[0][None, :]
        az = d["actions"][:, :, 2]
        thr = float(d["pick_lift_success_height"])
        for iid in NAMES:
            mask = ids == iid
            if not mask.any():
                continue
            grasped = held.any(0)
            lifted = (held & (lift >= thr)).any(0)
            tot[iid][0] += int(mask.sum())
            tot[iid][1] += int((mask & grasped).sum())
            tot[iid][2] += int((mask & lifted).sum())
            for w in np.flatnonzero(mask & grasped):
                k = held[:, w]
                acc[iid].append((az[k, w].mean(), lift[k, w].max()))
    print(f"== {arm}")
    for iid, name in NAMES.items():
        n, g, l = tot[iid]
        if not n or not acc[iid]:
            continue
        a = np.array(acc[iid])
        q = lambda c, p: float(np.percentile(a[:, c], p))  # noqa: E731
        print(
            f"  {name:8} n={n:4d} grasp={g/n:.4f} lift|grasp={l/max(g,1):.4f} "
            f"| a_z p10={q(0,10):+.3f} p50={q(0,50):+.3f} p90={q(0,90):+.3f} "
            f"| peak lift p50={q(1,50):.4f}"
        )
