# Why Physical AI Must Be Interpretable

Code, data pipeline and LaTeX source for the paper *Why Physical AI Must Be Interpretable* (Bhavith Chandra Challagundla, Elemental Research Group / NYU).

**Thesis.** A physical AI system that acts by consulting a learned world model can be trusted only to the extent that we can verify which variables the model represents and how those variables cause its predictions. Behavioral evaluation cannot provide that verification. The paper argues this formally, in three propositions:

1. Accuracy on the training distribution does not identify the mechanism a model learned.
2. A planner optimizing against a world model selects the model's errors.
3. Physical failures are rare and irreversible.

It then demonstrates the argument on pixel-based world models.

**Main empirical result.** Two pendulum world models are trained, one on data where bob color is confounded with gravity and one on causal data. The confounded model is *more accurate* on validation data, so validation-based model selection picks it in every seed. It fails once the confound breaks. Probes and interventions expose it using training data only. Exact numbers, per seed, are in [`VALIDATION.md`](VALIDATION.md) and in the PDF.

## Repository layout

```
main.tex, refs.bib        paper source         numbers.tex   all paper numbers (GENERATED, do not edit)
main.pdf                  compiled paper       fig_*.pdf     figures (GENERATED)
code/wm.py                simulator, renderer, world model, training
code/run.py               train one (condition, seed) and evaluate: behavior, probes, interventions, monitors
code/posthoc.py           latent steering, probe geometry
code/triangulate.py       independent re-derivation of every headline claim
code/aggregate.py         results/*.json  ->  numbers.tex + figures
code/validate.py          executable claims  ->  VALIDATION.md  (non-zero exit on any failure)
code/results/             per-seed metrics (JSON), checkpoints (.pt), training logs
code/results_unseeded_init/  an earlier full run with unseeded weight init (kept as an independent replication)
```

## Reproduce

```bash
pip install -r requirements.txt
make validate   # ~3 min: re-check every claim against committed results; regenerate numbers.tex and diff
make all        # ~60 min on 2 CPU cores: retrain 5 seeds x 2 conditions, analyze, validate, build PDF
```

No GPU is needed. How determinism, provenance, validation and triangulation are enforced is described in [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).

## Citation

See [`CITATION.cff`](CITATION.cff). Licensed under Apache-2.0.
