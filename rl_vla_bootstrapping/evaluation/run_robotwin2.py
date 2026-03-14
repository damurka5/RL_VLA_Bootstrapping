from __future__ import annotations

from rl_vla_bootstrapping.evaluation.delegate import parse_common_args, run_delegate


def main() -> int:
    args = parse_common_args("rl-vla-robotwin2", "RoboTwin 2.0")
    return run_delegate(args, "RoboTwin 2.0")


if __name__ == "__main__":
    raise SystemExit(main())
