# Claim validation

Seeds: [0, 1, 2, 3, 4]. Each claim must hold in **every** seed.

| ID | Claim | Status | Per-seed value |
|---|---|---|---|
| C1 | Validation error: confounded < causal (validation selects the shortcut) | PASS | s0=0.179, s1=0.195, s2=0.222, s3=0.176, s4=0.163 |
| C2 | Decorrelation raises confounded error > 3x | PASS | s0=7.79, s1=7.11, s2=5.98, s3=7.03, s4=7.93 |
| C3 | Causal model unaffected by decorrelation (ratio in [0.8,1.25]) | PASS | s0=0.979, s1=0.992, s2=0.99, s3=0.929, s4=1.02 |
| C4 | do(colour) shift: confounded > 5x causal | PASS | s0=13.5, s1=12.4, s2=51.6, s3=19.2, s4=25.9 |
| C5 | Confounded gravity probe R2 > 0.9 matched, < 0 decorrelated | PASS | s0=0.99/-0.97, s1=0.97/-0.96, s2=0.97/-0.73, s3=0.97/-0.96, s4=0.98/-0.99 |
| C6 | Causal gravity probe R2 < 0.1 (no linear gravity variable) | PASS | s0=0.0299, s1=0.0212, s2=0.031, s3=0.00503, s4=0.035 |
| C7 | cos(gravity probe, colour probe) > 0.9 in confounded | PASS | s0=0.997, s1=0.999, s2=0.999, s3=0.996, s4=0.996 |
| C8 | Latent steering on colour direction <= random direction (confounded) | PASS | s0=0.932, s1=0.428, s2=0.212, s3=0.71, s4=0.672 |
| C9 | Pre-action monitors near chance on causal/OOD-g (max AUROC < 0.65) | PASS | s0=0.582, s1=0.56, s2=0.596, s3=0.596, s4=0.65 |
| T1 | Error ordering holds under independent argmax readout | PASS | s0=1, s1=1, s2=1, s3=1, s4=1 |
| T2 | Causal model within 25% of 'knows state, g=mean' oracle | PASS | s0=1.12, s1=1.13, s2=1.11, s3=1.12, s4=1.14 |
| T2b | Confounded/decorr within 25% of 'g read from colour' oracle | PASS | s0=1.14, s1=1.11, s2=1.12, s3=1.12, s4=1.11 |
| T3 | Gravity not recoverable from causal latent by nonlinear probes (MLP, kNN R2 < 0.1) | PASS | s0=0.0186, s1=-0.00351, s2=0.021, s3=-0.00668, s4=0.0287 |
| T3b | Perfect sensor identifies gravity (R2 > 0.99); pixel readout does not (R2 < 0.2) | PASS | s0=1.000/-0.050, s1=1.000/-0.212, s2=1.000/-0.089, s3=1.000/-0.072, s4=1.000/-0.043 |
| T4 | Shortcut effect grows monotonically with colour dose | PASS | s0=1, s1=1, s2=1, s3=1, s4=1 |
| T5 | Bootstrap 95% CIs separate validation and decorrelation gaps | PASS | s0=1, s1=1, s2=1, s3=1, s4=1 |
| C10 | Planning optimism increases with N for both models (N=1 -> N=max, every seed) | PASS | s0=0.142, s1=0.138, s2=0.0828, s3=0.0992, s4=0.105 |
| C11 | Under planning, shortcut model realizes higher cost than causal for N in 2..16 | PASS | s0=0.0153, s1=0.0151, s2=0.0129, s3=0.00692, s4=0.0155 |
| C12 | Monte Carlo optimism matches closed form sigma*E[max] within 1% (Gaussian errors) | PASS | s0=0.00718 |
| D1 | Bit-level rerun of confounded s0 reproduces all metrics (max |diff| < 1e-6) | PASS | s0=0 |
| P1 | numbers.tex regenerated from results is identical to the committed one | PASS | s0=1 |

**21/21 claims pass.**
