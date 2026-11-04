# Matched development evaluation and preliminary WidowX results

Updated **4 November 2026**. This revision reports the separately evaluated **69.14% strict CDPR
result, 1,416/2,048 attempts**. It updates the abstract, experimental protocol,
result paragraph/table, scope discussion and conclusion. The 512 development
scenes each have four rollout repeats and exclude the fixed 128-scene training
panel. The reserved final-test split is unused; repeats are not training seeds.

The paired gain is **+8.35 pp**, CI **[+6.05, +10.69]**, Holm p=0.00005,
against the contemporaneous 40M baseline, 1,245/2,048 = 60.79%. Updated grasp,
lift, destination and failure rates all come from this same comparison. The
previous 61.43% baseline score is historical and is not used for this gain.
Intervals and p-values are retained as supplied; raw per-episode evaluations
are needed for independent recomputation.

The paper now includes the completed preliminary **WidowX-200 simulation**
evaluation: reaching 33.30%, pickup 24.02%, caught placement 35.64%, aligned
empty-hand composition 7.13%. The composed reset omits a distant approach,
so it is kept separate from CDPR strict completion. The discussion records
weak retention and the lack of demonstrated cross-embodiment reliability.
The former placeholder subsection is replaced with results, without adding
the placeholder apparatus figure or making a physical-robot claim.

Supporting records now include the supplied matched comparison and checked
counts, the WidowX final summary and snapshot audit, a separate WidowX CSV,
an updated CDPR inventory, source/missing-data notes and the results plan.
The project README leads with the off-panel result and updates the port's
implementation status and unresolved problems. LIBERO measurements, the
historical skill matrix and the three figures retain their existing data.

The pre-update reading copy and edited sources are preserved at
`archive/manuscript_20261104_before_matched_evaluation/`. The forced page break
before the bibliography was removed so references follow the article normally.
No training or evaluation was launched for this writing revision.

Validation: rebuilt with `latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex`. The current PDF has **six pages, three figures and three tables**. All pages were rendered and visually inspected, with detailed checks of the abstract, evaluation protocol and result pages. There are no overfull boxes or unresolved references. PDF text contains the 69.14% result. Evidence copies match their source bytes, inventory counts agree with the JSON, local links resolve, and `git diff --check` passes.
