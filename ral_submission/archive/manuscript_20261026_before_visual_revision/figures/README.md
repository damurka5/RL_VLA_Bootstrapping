# Manuscript figures

- `cdpr_training.pdf` / `.png`: generated from `evidence/latent_control_40m_validation.csv`; 81 unsmoothed readings of one training lineage. The x axis is cumulative selected actions since latent conversion, not all interactions or from-scratch cost.
- `cdpr_diagnostics.pdf` / `.png`: generated from the 19 October matched comparison; milestone rates and target-y quartiles. No inferred error bars or fabricated episode records.
- The former two-branch overview figure was removed on 22 October: it obscured the main adaptation sequence and could imply shared experiments. Figure numbers now follow the actual simulation plots and hardware slots.

Rebuild with `python3 scripts/generate_figures.py` in this folder's parent (`ral_submission/`); matplotlib is required. Included vector PDFs compile without Python.

## Hardware insertion slots

Put real images at these exact paths:

1. `hardware/setup.jpg`: actual robot, cable routing, end effector, table, objects/receptacles, cameras; add a scale if useful.
2. `hardware/sequence.jpg`: horizontal montage of 5 recorded frames (approach, grasp, held lift, transport, release); add relative timestamps and panel letters.

The LaTeX automatically replaces the placeholder boxes when these files exist. Their dimensions are reserved, so later insertion is simple; adjust height if needed and check readability after compilation. Do not use generated images as experimental evidence. Keep full-resolution originals separately.

Attach instruction, checkpoint, trial ID, object/destination, camera viewpoint and deployment condition to each sequence. Supply a representative failure sequence/video and the corresponding measured frequency. Final captions must describe the actual frames; current captions explicitly identify reserved content.

## Next figure requiring new data

LIBERO learning and throughput: validation versus wall time plus accepted/attempted rollout groups and optimizer updates versus wall time. The source PDF contains only a qualitative faster-collection claim. Do not draw a numerical throughput chart without the logs.
