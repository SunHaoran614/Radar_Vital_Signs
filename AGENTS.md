# Repository instructions

## Scope and layout

- Maintain reusable runtime implementations under `src/`, reusable configurations under `configs/`, and self-contained tests under `tests/`.
- Keep public usage documentation under `docs/`. User-reviewed paper figures belong in `docs/manuscript/figures/reviewed/`; replacing a reviewed version requires the user's review.
- All historical experiments, results, audits, one-off scripts, manuscripts and replay snapshots belong in the local `historical_archive/` directory. Do not stage, package or upload this directory.
- Runtime code and public tests must not import from the archive or historical `scripts`/`experiments` modules. Extract reusable implementations into `src/` and verify compatibility before archiving their callers.
- Do not delete archived evidence or alter frozen artifact bytes. Original research instructions and protocols are preserved in `historical_archive/repository/AGENTS.md` and its `docs/` tree for local replay.
- Do not stage raw data, weights, predictions, logs, private configuration or generated manuscripts. Never rewrite Git history without an explicit request.

## Research invariants

- The user-approved second research point deliverables are 8 main tables and 2 appendices, recorded locally in `historical_archive/SECOND_RESEARCH_TABLE_PLAN.md`. Consult this checklist before organizing thesis results; do not silently reduce its scope or upload the local plan.

- The complete research chain is frontend processing → deep HR prediction → cross-domain evaluation. Frontend diagnostics or shallow probes alone do not prove the complete method.
- Training and validation run locally, not on Kaggle.
- FTU is the labeled source in the FTU protocol. Use source subject groups and remove confirmed physical-record aliases. Select on train, review val only after fixing candidates, and evaluate test only after freezing the method.
- Nature may be a separate labeled source under its independent protocol. Target HR/ECG labels must never enter source training, candidate selection or unlabeled adaptation.
- Nature provides complex range FFT, not raw ADC. PhysDrive reference timing and HR quality remain uncertified. Do not use these data to claim validated raw ADC frontends or certified HR labels.
- Already-scored test sets cannot tune the same model, frontend or adaptation version. Do not reclassify consumed records as a new independent sealed test.
- Report absolute BPM error and within-record changes for the same prediction, with no-radar source baselines and subject-level statistics. Lag k pairs must belong to the same physical record and have original window IDs differing by exactly k; report pair counts and the zero-change baseline on the same pairs.
- Do not select clock/label corrections using model errors or convert exploratory, oracle-calibrated or target-selected results into deployment evidence.

## Verification

Run `python -m pytest -q` and `python scripts/check_release.py` after structural changes. Packaging must explicitly include only maintained code; the archive must remain excluded even if present locally. Do not run training, scoring or label regeneration as an incidental repository check.
