# Benchmarks

This repo includes evaluation adapters for RoboTwin 2.0 and ManiTask, but not the benchmark repos themselves.
GitHub keeps the adapters plus placeholder external directories; the actual benchmark repos are staged locally.

Stage benchmark workspaces into:

- `benchmarks/externals/robotwin2`
- `benchmarks/externals/manitask`

Then enable the matching benchmark block in the config and set the real benchmark entry script if it differs from the placeholder `run_benchmark.py`.
