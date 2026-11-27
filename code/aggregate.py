"""Aggregate per-seed JSON -> numbers.tex (LaTeX macros) + figures. Every number in the paper flows through here."""
import json, glob, os, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import re as _re
SEEDS = sorted(int(_re.search(r"_s(\d+)\.json", f).group(1)) for f in glob.glob("results/clean_s*.json") if os.path.exists(f.replace("clean", "confounded")))
R = {m: [json.load(open(f"results/{m}_s{s}.json")) for s in SEEDS] for m in ["clean", "confounded"]}
PH = json.load(open("results/posthoc.json"))
macros = {}
def ms(vals, nd=3):
    v = np.array(vals, float); return f"{v.mean():.{nd}f}\\,$\\pm$\\,{v.std(ddof=1):.{nd}f}", v.mean()
def put(name, vals, nd=3):
    s, mean = ms(vals, nd); macros[name] = s; macros[name + "Mean"] = f"{mean:.{nd}f}"; return mean
tag = {"clean": "Cl", "confounded": "Co"}
for m, rs in R.items():
    t = tag[m]
    for split, key in [("val_matched", "Val"), ("decorrelated", "Dec"), ("ood_gravity", "OodG"), ("ood_damping", "OodD")]:
        put(f"err{t}{key}", [r[split]["angle_err_hH"] for r in rs])
        put(f"mse{t}{key}", [r[split]["pixel_mse_x1e3"] for r in rs], 2)
    for k in ["sin_theta", "cos_theta", "omega", "gravity", "damping", "redness"]:
        for split, key in [("val_matched", "Val"), ("decorrelated", "Dec")]:
            put(f"rtwo{t}{k.replace('_','')}{key}", [r["probe_r2"][k][split] for r in rs], 2)
    put(f"intShift{t}", [r["intervention"]["mean_abs_angle_shift_hH"] for r in rs])
    put(f"intGimp{t}", [r["intervention"]["implied_g_shift_per_unit_redness"] for r in rs], 2)
    put(f"intGprobe{t}", [r["intervention"]["probe_g_shift_per_unit_redness"] for r in rs], 2)
    put(f"cosGR{t}", [PH[f"{m}_s{s}"]["cos_gravity_vs_colour_probe"] for s in SEEDS], 2)
    put(f"steerC{t}", [PH[f"{m}_s{s}"]["steer_colour_angle_shift_hH"] for s in SEEDS])
    put(f"steerR{t}", [PH[f"{m}_s{s}"]["steer_random_angle_shift_hH"] for s in SEEDS])
    for shift, key in [("ood_gravity", "G"), ("ood_damping", "D"), ("decorrelated", "Dec")]:
        for sc, sk in [("pixel_knn", "Knn"), ("latent_mahalanobis", "Maha"), ("probe_physics_range", "Prange"),
                       ("kinematic_consistency", "Kin"), ("lagged_1step_residual", "Lag")]:
            put(f"auc{t}{key}{sk}", [r["monitor"][shift][sc]["auroc_shift"] for r in rs], 2)
macros["ratioDecVal"] = f"{np.mean([r['decorrelated']['angle_err_hH'] for r in R['confounded']]) / np.mean([r['val_matched']['angle_err_hH'] for r in R['confounded']]):.1f}"
macros["ratioValClCo"] = f"{float(macros['errClValMean'])/float(macros['errCoValMean']):.1f}"
macros["ratioIntCoCl"] = f"{float(macros['intShiftCoMean'])/float(macros['intShiftClMean']):.0f}"
macros["nSeeds"] = str(len(SEEDS))
macros["nTrainWin"] = str(R["clean"][0]["n_train_windows"]); macros["nEpochs"] = str(R["clean"][0]["epochs"])
# per-seed shortcut-wins check: does validation select the confounded model in every seed?
macros["valPicksShortcut"] = str(sum(c["val_matched"]["angle_err_hH"] > k["val_matched"]["angle_err_hH"] for c, k in zip(R["clean"], R["confounded"])))

# ---- triangulation macros (independent re-derivations)
if os.path.exists("results/triangulation.json"):
    TR = json.load(open("results/triangulation.json")); TS = [TR[f"s{s}"] for s in SEEDS if f"s{s}" in TR]
    put("triArgClVal", [t["T1_readout"]["argmax_err_hH"]["clean_val"] for t in TS])
    put("triArgCoVal", [t["T1_readout"]["argmax_err_hH"]["conf_val"] for t in TS])
    put("triArgCoDec", [t["T1_readout"]["argmax_err_hH"]["conf_decorr"] for t in TS])
    put("oraMeanG", [t["T2_oracles_err_hH"]["knows_state_g_fixed_at_mean_10"] for t in TS])
    put("oraColG", [t["T2_oracles_err_hH"]["g_read_from_colour_6+8r"] for t in TS])
    put("oraTrueG", [t["T2_oracles_err_hH"]["true_g"] for t in TS])
    put("oraModelCl", [t["T2_oracles_err_hH"]["model_clean"] for t in TS])
    put("oraModelCo", [t["T2_oracles_err_hH"]["model_confounded"] for t in TS])
    put("idMlp", [t["T3_gravity_identifiability"]["mlp_probe_r2"] for t in TS], 2)
    put("idKnn", [t["T3_gravity_identifiability"]["knn_probe_r2"] for t in TS], 2)
    put("idExact", [t["T3_gravity_identifiability"]["exact_angles_r2_(perfect_sensor)"] for t in TS], 3)
    put("idPixel", [t["T3_gravity_identifiability"]["physics_from_3_frames_r2_(|sin|>0.15)"] for t in TS], 2)
    put("readErrMean", [1e3*t["T3_gravity_identifiability"]["centroid_readout_err_mean_rad"] for t in TS], 2)
    put("readErrMax", [1e3*t["T3_gravity_identifiability"]["centroid_readout_err_max_rad"] for t in TS], 1)
    for i, d in enumerate([5, 10, 20, 30]):
        put(f"dose{['A','B','C','D'][i]}Co", [t["T4_dose_response"]["conf"][i] for t in TS])
        put(f"dose{['A','B','C','D'][i]}Cl", [t["T4_dose_response"]["clean"][i] for t in TS])
with open("../numbers.tex", "w") as f:
    for k, v in macros.items():
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
json.dump(macros, open("results/macros.json", "w"), indent=1)

# ---------- figures
plt.rcParams.update({"font.size": 9, "font.family": "serif", "mathtext.fontset": "cm", "axes.spines.top": False, "axes.spines.right": False})
C = {"clean": "#3b6ea8", "confounded": "#c0504d"}
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.3))
splits = [("val_matched", "matched"), ("decorrelated", "decorr."), ("ood_gravity", "OOD $g$")]
x = np.arange(len(splits)); w = 0.38
for i, m in enumerate(["clean", "confounded"]):
    v = np.array([[r[s]["angle_err_hH"] for s, _ in splits] for r in R[m]])
    ax[0].bar(x + (i-0.5)*w, v.mean(0), w, yerr=v.std(0, ddof=1), color=C[m], label="causal-data model" if m == "clean" else "confounded-data model", capsize=2)
ax[0].set_xticks(x); ax[0].set_xticklabels([l for _, l in splits]); ax[0].set_ylabel(r"angle error at $t{+}H$ (rad)")
ax[0].set_title("(a) Behaviour", fontsize=9); ax[0].set_ylim(0, 0.62); ax[0].legend(frameon=False, fontsize=6.5, loc="upper left", ncol=1)
for i, m in enumerate(["clean", "confounded"]):
    v = np.array([r["intervention"]["mean_abs_angle_shift_hH"] for r in R[m]])
    ax[1].bar(i, v.mean(), 0.6, yerr=v.std(ddof=1), color=C[m], capsize=2)
ax[1].set_xticks([0, 1]); ax[1].set_xticklabels(["causal", "confounded"]); ax[1].set_ylabel("prediction shift (rad)")
ax[1].set_title("(b) do(color) intervention", fontsize=9)
for i, m in enumerate(["clean", "confounded"]):
    v = np.array([r["probe_r2"]["gravity"]["val_matched"] for r in R[m]]); v2 = np.array([r["probe_r2"]["gravity"]["decorrelated"] for r in R[m]])
    ax[2].bar(i - 0.2, v.mean(), 0.38, yerr=v.std(ddof=1), color=C[m], capsize=2)
    ax[2].bar(i + 0.2, v2.mean(), 0.38, yerr=v2.std(ddof=1), color=C[m], alpha=0.45, hatch="//", capsize=2)
ax[2].axhline(0, color="k", lw=0.6); ax[2].set_xticks([0, 1]); ax[2].set_xticklabels(["causal", "confounded"])
ax[2].set_ylabel(r"gravity probe $R^2$"); ax[2].set_title("(c) Probe: matched / decorrelated", fontsize=9)
plt.tight_layout(); plt.savefig("../fig_main.pdf", bbox_inches="tight")

fig, ax = plt.subplots(figsize=(3.4, 2.7))
scs = [("pixel_knn", "pixel kNN"), ("latent_mahalanobis", "latent Mahalanobis"), ("probe_physics_range", "probe: $g$ range"),
       ("kinematic_consistency", "probe: kinematic"), ("lagged_1step_residual", "lagged residual$^\\dagger$")]
groups = [("clean", "ood_gravity", "causal / OOD $g$"), ("clean", "ood_damping", "causal / OOD damping"),
          ("confounded", "decorrelated", "confounded / decorr.")]
y = np.arange(len(scs))[::-1]; w = 0.26; cols = ["#3b6ea8", "#7fa6d3", "#c0504d"]
for j, (m, sh, lab) in enumerate(groups):
    v = np.array([[r["monitor"][sh][k]["auroc_shift"] for k, _ in scs] for r in R[m]])
    ax.barh(y - (j-1)*w, v.mean(0), w, xerr=v.std(0, ddof=1), color=cols[j], label=lab, capsize=1.5)
ax.axvline(0.5, color="k", ls="--", lw=0.7); ax.set_xlim(0.3, 0.8); ax.set_yticks(y); ax.set_yticklabels([l for _, l in scs])
ax.set_xlabel("AUROC (shifted vs. matched)"); ax.legend(frameon=False, fontsize=6.5, loc="lower center", bbox_to_anchor=(0.35, 1.0), ncol=2)
plt.tight_layout(); plt.savefig("../fig_monitor.pdf", bbox_inches="tight")
print(json.dumps(macros, indent=0))
