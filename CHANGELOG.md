# Changelog

Bugs found by the validation and triangulation suite are recorded here, together with how each one affected the reported results.

## v1.2 (2026-11-27)
- The primary paper (`main.tex`) is now a single-column, argument-led preprint in the style of LeCun's position papers. Every section opens with theory: an energy-based framing of planning; a new Proposition (off-support exploitation); formal definitions of representation and causal use.
- New diagrams: an energy landscape showing how a planner finds model errors, and a ladder of evidence mapped to World Model Lens primitives.
- Cites World Model Lens (Challagundla, 2026) as tooling. The paper states explicitly that the experiments use a self-contained harness rather than WML.
- The NeurIPS-format version is kept as `main_neurips.tex`.

## v1.1 (2026-11-27)
- NeurIPS-format rewrite, with new figures (teaser of causal graphs, agent architecture, model inputs, planner).
- **New experiment:** a planner using the trained models confirms Proposition 3. At N=64, realized cost is about 12x the predicted cost. New claims C10-C12; 21/21 pass.
- The shortcut model is worse under planning for N=2..16 in every seed; at N=32-64 one seed reverses, so the claim is scoped to N<=16.

## v1.0 (2026-11-27)

### Fixed
- **Weight initialization was not seeded.** `torch.manual_seed` was called inside `train()`, after `WorldModel()` had already initialized its weights. Batch order was reproducible, but initial weights were not, and the determinism check (D1) failed: a from-scratch rerun diverged at epoch 1. Fixed by seeding before model construction in `run.py`, then retraining all runs. The earlier run is kept in `code/results_unseeded_init/` as an independent replication under random initialization.
- **Angle-readout accuracy was overstated.** An early draft stated that the centroid readout recovers angles to within 1e-6 rad. That was true before a 50%-of-peak threshold was added to the readout. The measured error after the change is a few milliradians (triangulation T3), and the paper now reports the measured value.

### Validation status
- 18/18 executable claims pass in all 5 seeds. The tightest margin is C9 on seed 4: the best pre-action monitor AUROC is 0.6499 against a 0.65 threshold.
- The unseeded-initialization archive independently reproduces the headline ratios in each of its 5 seeds: validation error ratio 0.17–0.21, decorrelation increase 6.1–7.6×, intervention effect 10–14×.

### Found by triangulation
- **Gravity is barely identifiable from three 32x32 frames.** With exact simulator angles, a finite-difference estimator recovers gravity almost perfectly (R² > 0.99). With angles read from pixels, it fails (R² ≈ 0), because double differencing amplifies millirad readout error by about 1/Δt². The causal model's lack of a gravity variable is therefore mostly a limit of the sensor, not a defect of the model. It also explains why the color shortcut is preferred: color is the only cheap route to gravity. The paper's interpretation was updated accordingly.

### Pilot-stage fixes (before any reported run)
- Sigmoid decoder saturated to an all-zero output. Replaced with a linear output, gradient clipping and a lower learning rate.
- Decoder collapsed to an empty frame because the bob covers about 1% of the pixels. Added a foreground-weighted loss.
- A prediction horizon of 0.3 s gave gravity too little influence on the trajectory to test the shortcut. Changed to Δt = 0.08 and H = 8 (0.64 s).
