from __future__ import annotations

from rl_vla_bootstrapping.evaluation.delegate import parse_common_args, run_delegate


def main() -> int:
    args = parse_common_args("rl-vla-manitask", "ManiTask")
    return run_delegate(args, "ManiTask")


if __name__ == "__main__":
    raise SystemExit(main())
