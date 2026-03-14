from __future__ import annotations

import argparse
import sys

from rl_vla_bootstrapping.pipeline.bootstrap import BootstrapPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="rl-vla-bootstrap-train",
        description="Plan or execute an embodiment-first RL -> SFT bootstrap pipeline.",
    )
    parser.add_argument("--config", required=True, help="Path to YAML/JSON/TOML project config.")
    parser.add_argument(
        "--stage",
        action="append",
        default=[],
        help="Stage to include: preview, rl, sft, eval, or all. Can be repeated.",
    )
    parser.add_argument("--run-name", default=None, help="Optional run directory name.")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute the planned stages. Without this flag, only print the plan.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pipeline = BootstrapPipeline.from_path(args.config)
    requested_stages = args.stage or ["all"]

    errors = pipeline.validate()
    if errors:
        for error in errors:
            print(f"[config-error] {error}", file=sys.stderr)
        return 2

    run_dir = pipeline.make_run_dir(run_name=args.run_name)
    manifests = pipeline.export_manifests(run_dir)
    plans = pipeline.build_stage_plans(run_dir, requested_stages)

    print(f"Run directory: {run_dir}")
    for key, path in manifests.items():
        print(f"Manifest `{key}`: {path}")

    for plan in plans:
        print(f"\n[{plan.name}] {plan.kind}")
        for note in plan.notes:
            print(f"  note: {note}")
        if plan.command:
            print(f"  cwd: {plan.cwd}")
            print(f"  cmd: {plan.pretty_command()}")
        for artifact in plan.artifact_paths:
            print(f"  artifact: {artifact}")

    if not args.execute:
        return 0

    try:
        results = pipeline.execute(run_dir, requested_stages)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    failed = [(plan, code) for plan, code in results if code != 0]
    if failed:
        plan, code = failed[0]
        print(f"\nStage `{plan.name}` failed with exit code {code}.", file=sys.stderr)
        return int(code)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
