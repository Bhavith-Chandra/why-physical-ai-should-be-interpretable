"""Claim-level validation. Every headline claim in the paper is an executable check over ALL seeds present.
Writes VALIDATION.md and exits non-zero if any claim fails. Run after run.py/posthoc.py/triangulate.py/aggregate.py."""
import json, glob, os, re, subprocess, sys, numpy as np

seeds = sorted(int(re.search(r"_s(\d+)\.json", f).group(1)) for f in glob.glob("results/clean_s*.json")
               if os.path.exists(f.replace("clean", "confounded")))
R = {m: {s: json.load(open(f"results/{m}_s{s}.json")) for s in seeds} for m in ["clean", "confounded"]}
PH = json.load(open("results/posthoc.json")) if os.path.exists("results/posthoc.json") else {}
TR = json.load(open("results/triangulation.json")) if os.path.exists("results/triangulation.json") else {}
rows, failed = [], 0

def check(cid, claim, per_seed, rule):
    """per_seed: {seed: value}; rule: value -> bool. Claim must hold in EVERY seed."""
    global failed
    ok = all(rule(v) for v in per_seed.values()) and len(per_seed) > 0
    failed += (not ok)
    vals = ", ".join(f"s{s}={v:.3g}" if isinstance(v, (int, float)) else f"s{s}={v}" for s, v in per_seed.items())
    rows.append(f"| {cid} | {claim} | {'PASS' if ok else '**FAIL**'} | {vals} |")

c, k = R["clean"], R["confounded"]
check("C1", "Validation error: confounded < causal (validation selects the shortcut)",
      {s: k[s]["val_matched"]["angle_err_hH"] / c[s]["val_matched"]["angle_err_hH"] for s in seeds}, lambda v: v < 1)
check("C2", "Decorrelation raises confounded error > 3x", {s: k[s]["decorrelated"]["angle_err_hH"] / k[s]["val_matched"]["angle_err_hH"] for s in seeds}, lambda v: v > 3)
check("C3", "Causal model unaffected by decorrelation (ratio in [0.8,1.25])", {s: c[s]["decorrelated"]["angle_err_hH"] / c[s]["val_matched"]["angle_err_hH"] for s in seeds}, lambda v: 0.8 < v < 1.25)
check("C4", "do(colour) shift: confounded > 5x causal", {s: k[s]["intervention"]["mean_abs_angle_shift_hH"] / c[s]["intervention"]["mean_abs_angle_shift_hH"] for s in seeds}, lambda v: v > 5)
check("C5", "Confounded gravity probe R2 > 0.9 matched, < 0 decorrelated", {s: f"{k[s]['probe_r2']['gravity']['val_matched']:.2f}/{k[s]['probe_r2']['gravity']['decorrelated']:.2f}" for s in seeds},
      lambda v: float(v.split('/')[0]) > 0.9 and float(v.split('/')[1]) < 0)
check("C6", "Causal gravity probe R2 < 0.1 (no linear gravity variable)", {s: c[s]["probe_r2"]["gravity"]["val_matched"] for s in seeds}, lambda v: v < 0.1)
if PH:
    check("C7", "cos(gravity probe, colour probe) > 0.9 in confounded", {s: PH[f"confounded_s{s}"]["cos_gravity_vs_colour_probe"] for s in seeds if f"confounded_s{s}" in PH}, lambda v: v > 0.9)
    check("C8", "Latent steering on colour direction <= random direction (confounded)", {s: PH[f"confounded_s{s}"]["steer_colour_angle_shift_hH"] / PH[f"confounded_s{s}"]["steer_random_angle_shift_hH"] for s in seeds if f"confounded_s{s}" in PH}, lambda v: v <= 1.0)
check("C9", "Pre-action monitors near chance on causal/OOD-g (max AUROC < 0.65)",
      {s: max(v["auroc_shift"] for kk, v in c[s]["monitor"]["ood_gravity"].items() if kk != "lagged_1step_residual") for s in seeds}, lambda v: v < 0.65)
if TR:
    T = {int(s[1:]): v for s, v in TR.items()}
    check("T1", "Error ordering holds under independent argmax readout", {s: v["T1_readout"]["ordering_holds"] for s, v in T.items()}, lambda v: v is True)
    check("T2", "Causal model within 25% of 'knows state, g=mean' oracle", {s: v["T2_oracles_err_hH"]["model_clean"] / v["T2_oracles_err_hH"]["knows_state_g_fixed_at_mean_10"] for s, v in T.items()}, lambda v: 0.75 < v < 1.25)
    check("T2b", "Confounded/decorr within 25% of 'g read from colour' oracle", {s: v["T2_oracles_err_hH"]["model_confounded"] / v["T2_oracles_err_hH"]["g_read_from_colour_6+8r"] for s, v in T.items()}, lambda v: 0.75 < v < 1.25)
    check("T3", "Gravity not recoverable from causal latent by nonlinear probes (MLP, kNN R2 < 0.1)", {s: max(v["T3_gravity_identifiability"]["mlp_probe_r2"], v["T3_gravity_identifiability"]["knn_probe_r2"]) for s, v in T.items()}, lambda v: v < 0.1)
    check("T3b", "Perfect sensor identifies gravity (R2 > 0.99); pixel readout does not (R2 < 0.2)", {s: f"{v['T3_gravity_identifiability']['exact_angles_r2_(perfect_sensor)']:.3f}/{v['T3_gravity_identifiability']['physics_from_3_frames_r2_(|sin|>0.15)']:.3f}" for s, v in T.items()},
          lambda v: float(v.split('/')[0]) > 0.99 and float(v.split('/')[1]) < 0.2)
    check("T4", "Shortcut effect grows monotonically with colour dose", {s: v["T4_dose_response"]["conf_monotone"] for s, v in T.items()}, lambda v: v is True)
    check("T5", "Bootstrap 95% CIs separate validation and decorrelation gaps", {s: v["T5_bootstrap95"]["val_gap_excludes_0"] and v["T5_bootstrap95"]["decorr_gap_excludes_0"] for s, v in T.items()}, lambda v: v is True)

if os.path.exists("results/planner.json"):
    P = json.load(open("results/planner.json")); N = P["N"]
    check("C10", "Planning optimism increases with N for both models (N=1 -> N=max, every seed)",
          {int(s): min(P["seeds"][s][m][str(N[-1])]["optimism"] - P["seeds"][s][m]["1"]["optimism"] for m in ["clean", "confounded"]) for s in P["seeds"]}, lambda v: v > 0.05)
    check("C11", "Under planning, shortcut model realizes higher cost than causal for N in 2..16",
          {int(s): min(P["seeds"][s]["confounded"][str(n)]["realized"] - P["seeds"][s]["clean"][str(n)]["realized"] for n in [2, 4, 8, 16]) for s in P["seeds"]}, lambda v: v > 0)
    G = P["gaussian_check"]
    check("C12", "Monte Carlo optimism matches closed form sigma*E[max] within 1% (Gaussian errors)",
          {0: max(abs(G[k]["monte_carlo"] - G[k]["exact_E_max"]) / max(G[k]["exact_E_max"], 1e-9) for k in G if k != "1")}, lambda v: v < 0.01)

# determinism: an independent re-run of confounded seed 0 must reproduce the stored metrics
if os.path.exists("rerun/confounded_s0.json"):
    a, b = json.load(open("results/confounded_s0.json")), json.load(open("rerun/confounded_s0.json"))
    def flat(d, p=""):
        for kk, v in d.items():
            if isinstance(v, dict): yield from flat(v, p + kk + ".")
            elif isinstance(v, float): yield p + kk, v
    fa, fb = dict(flat(a)), dict(flat(b))
    diff = max(abs(fa[x] - fb[x]) for x in fa)
    check("D1", "Bit-level rerun of confounded s0 reproduces all metrics (max |diff| < 1e-6)", {0: diff}, lambda v: v < 1e-6)

# paper numbers come only from aggregate.py: regenerate and diff
before = open("../numbers.tex").read() if os.path.exists("../numbers.tex") else ""
subprocess.run([sys.executable, "aggregate.py"], check=True, capture_output=True)
subprocess.run([sys.executable, "figures.py"], check=True, capture_output=True)
check("P1", "numbers.tex regenerated from results is identical to the committed one", {0: int(before == open("../numbers.tex").read())}, lambda v: v == 1)

with open("../VALIDATION.md", "w") as f:
    f.write(f"# Claim validation\n\nSeeds: {seeds}. Each claim must hold in **every** seed.\n\n| ID | Claim | Status | Per-seed value |\n|---|---|---|---|\n")
    f.write("\n".join(rows) + f"\n\n**{len(rows)-failed}/{len(rows)} claims pass.**\n")
print("\n".join(rows)); print(f"{len(rows)-failed}/{len(rows)} pass")
sys.exit(1 if failed else 0)
