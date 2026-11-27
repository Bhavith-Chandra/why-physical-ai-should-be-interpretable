"""All data figures for the paper, drawn only from results/*.json (and the simulator for the frame panel).
NeurIPS text width = 5.5in. Typeface = TeX Gyre Termes (Times), matching the paper body."""
import json, glob, os, re, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from wm import dataset, simulate, render, make_colors, R

for f in glob.glob("/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyretermes-*.otf"): fm.fontManager.addfont(f)
plt.rcParams.update({
    "font.family": "TeX Gyre Termes", "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8.5,
    "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.grid": True, "grid.linewidth": 0.4, "grid.alpha": 0.35, "axes.axisbelow": True,
    "savefig.dpi": 300, "pdf.fonttype": 3, "legend.frameon": False})
W = 5.5
BLUE, RED, GREY, LBLUE = "#2f5d8c", "#b5443b", "#6b6b6b", "#8fb3d9"
OUT = ".."
seeds = sorted(int(re.search(r"clean_s(\d+)\.json", f).group(1)) for f in glob.glob("results/clean_s*.json"))
Rs = {m: [json.load(open(f"results/{m}_s{s}.json")) for s in seeds] for m in ["clean", "confounded"]}
def ms(a): a = np.asarray(a, float); return a.mean(0), a.std(0, ddof=1)
def label(ax, t): ax.set_title(t, loc="left", fontweight="bold", pad=4)

# ---------------------------------------------------------------- Fig: environment frames
def fig_frames():
    rng = np.random.default_rng(7)
    fig, axes = plt.subplots(2, 7, figsize=(W, 1.75), gridspec_kw={"wspace": 0.06, "hspace": 0.35})
    gs = np.linspace(6, 14, 7)
    for row, mode in enumerate(["confounded", "clean"]):
        col, _ = make_colors(gs, mode, rng)
        th = np.full((7, 1), 0.7)
        img = render(th, col)[:, 0]
        for j in range(7):
            ax = axes[row, j]; ax.imshow(np.transpose(img[j], (1, 2, 0)), interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for sp in ax.spines.values(): sp.set_visible(True); sp.set_linewidth(0.4); sp.set_color(GREY)
            if row == 0: ax.set_title(f"$g={gs[j]:.1f}$", fontsize=7, pad=2)
        axes[row, 0].set_ylabel("confounded\ntraining" if mode == "confounded" else "causal\ntraining", fontsize=7, rotation=0, ha="right", va="center", labelpad=4)
    fig.savefig(f"{OUT}/fig_frames.pdf", bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- Fig: main behavioral/interp results
def fig_main():
    fig, ax = plt.subplots(1, 3, figsize=(W, 1.75), gridspec_kw={"wspace": 0.55})
    splits = [("val_matched", "val."), ("decorrelated", "decorr."), ("ood_gravity", "OOD $g$")]
    x = np.arange(3); w = 0.36
    for i, (m, c, lab) in enumerate([("clean", BLUE, "causal data"), ("confounded", RED, "confounded data")]):
        mu, sd = ms([[r[s]["angle_err_hH"] for s, _ in splits] for r in Rs[m]])
        ax[0].bar(x + (i-0.5)*w, mu, w, yerr=sd, color=c, label=lab, error_kw={"elinewidth": 0.7, "capsize": 1.5})
    ax[0].set_xticks(x, [l for _, l in splits]); ax[0].set_ylabel("angle error at $t{+}H$ (rad)")
    ax[0].legend(loc="upper left", handlelength=1, borderaxespad=0); ax[0].set_ylim(0, 0.62); label(ax[0], "(a) behavior (L0)")
    for i, (m, c) in enumerate([("clean", BLUE), ("confounded", RED)]):
        mu, sd = ms([r["intervention"]["mean_abs_angle_shift_hH"] for r in Rs[m]])
        ax[1].bar(i, mu, 0.6, yerr=sd, color=c, error_kw={"elinewidth": 0.7, "capsize": 1.5})
    ax[1].set_xticks([0, 1], ["causal", "confounded"]); ax[1].set_ylabel("prediction shift (rad)"); label(ax[1], "(b) do(color) (L2)")
    for i, (m, c) in enumerate([("clean", BLUE), ("confounded", RED)]):
        for j, sp in enumerate(["val_matched", "decorrelated"]):
            mu, sd = ms([r["probe_r2"]["gravity"][sp] for r in Rs[m]])
            ax[2].bar(i + (j-0.5)*0.36, mu, 0.36, yerr=sd, color=c, alpha=1 if j == 0 else 0.4, hatch=None if j == 0 else "////",
                      edgecolor="white" if j == 0 else c, linewidth=0.0 if j == 0 else 0.5, error_kw={"elinewidth": 0.7, "capsize": 1.5})
    ax[2].axhline(0, color="k", lw=0.5); ax[2].set_xticks([0, 1], ["causal", "confounded"]); ax[2].set_ylabel("gravity probe $R^2$")
    ax[2].text(1.18, 0.08, "decorr.", fontsize=6.5, color=RED, ha="center"); ax[2].text(0.82, 1.03, "val.", fontsize=6.5, color=RED, ha="center")
    ax[2].set_ylim(-1.15, 1.2); label(ax[2], "(c) probe (L1)")
    fig.savefig(f"{OUT}/fig_main.pdf", bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- Fig: planner optimism (Prop. 3)
def fig_planner():
    P = json.load(open("results/planner.json")); N = P["N"]; S = list(P["seeds"])
    fig, ax = plt.subplots(1, 3, figsize=(W, 1.8), gridspec_kw={"wspace": 0.62})
    get = lambda m, k: ms([[P["seeds"][s][m][str(n)][k] for n in N] for s in S])
    for m, c, lab in [("clean", BLUE, "causal"), ("confounded", RED, "confounded")]:
        mu, sd = get(m, "realized"); ax[0].plot(N, mu, "-o", color=c, ms=2.5, lw=1, label=f"realized ({lab})"); ax[0].fill_between(N, mu-sd, mu+sd, color=c, alpha=0.15, lw=0)
        mu, sd = get(m, "predicted"); ax[0].plot(N, mu, "--", color=c, lw=0.9, label=f"predicted ({lab})")
    mu, _ = get("clean", "oracle_best"); ax[0].plot(N, mu, ":", color="k", lw=1, label="best achievable")
    ax[0].set_xscale("log", base=2); ax[0].set_yscale("log"); ax[0].set_xlabel("candidates compared $N$"); ax[0].set_ylabel("cost $|\\theta_{t+H}|$ (rad)")
    from matplotlib.ticker import FixedLocator, NullLocator
    ax[0].yaxis.set_major_locator(FixedLocator([0.01, 0.03, 0.1, 0.3])); ax[0].yaxis.set_minor_locator(NullLocator())
    ax[0].set_yticklabels(["0.01", "0.03", "0.1", "0.3"])
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], color=BLUE, lw=1.2), Line2D([], [], color=RED, lw=1.2), Line2D([], [], color=GREY, lw=1, ls="-"),
          Line2D([], [], color=GREY, lw=1, ls="--"), Line2D([], [], color="k", lw=1, ls=":")]
    fig.legend(hs, ["causal", "confounded", "realized", "predicted", "best achievable"], loc="upper center", ncol=5,
               bbox_to_anchor=(0.5, 1.1), handlelength=1.8, columnspacing=1.2)
    label(ax[0], "(a) predicted vs. realized")
    for m, c, lab in [("clean", BLUE, "causal"), ("confounded", RED, "confounded")]:
        mu, sd = get(m, "optimism"); ax[1].plot(N, mu, "-o", color=c, ms=2.5, lw=1, label=lab); ax[1].fill_between(N, mu-sd, mu+sd, color=c, alpha=0.15, lw=0)
    ax[1].axhline(0, color="k", lw=0.5); ax[1].set_xscale("log", base=2); ax[1].set_xlabel("candidates compared $N$"); ax[1].set_ylabel("realized $-$ predicted (rad)")
    label(ax[1], "(b) optimism")
    G = P["gaussian_check"]; n = np.array(sorted(int(k) for k in G))
    ax[2].plot(n, [G[str(k)]["exact_E_max"] for k in n], "-", color="k", lw=1, label="closed form")
    ax[2].plot(n, [G[str(k)]["monte_carlo"] for k in n], "o", color=RED, ms=3, mfc="none", mew=0.8, label="Monte Carlo")
    ax[2].set_xscale("log", base=2); ax[2].set_xlabel("candidates compared $N$"); ax[2].set_ylabel("optimism ($\\sigma$ units)")
    ax[2].legend(loc="lower right", handlelength=1.4); label(ax[2], "(c) theory check")
    fig.savefig(f"{OUT}/fig_planner.pdf", bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- Fig: monitors
def fig_monitor():
    scs = [("pixel_knn", "pixel $k$NN"), ("latent_mahalanobis", "latent Mahalanobis"), ("probe_physics_range", "probe: $g$ range"),
           ("kinematic_consistency", "probe: kinematic"), ("lagged_1step_residual", "lagged residual$^\\dagger$")]
    groups = [("clean", "ood_gravity", "causal / OOD $g$", BLUE), ("clean", "ood_damping", "causal / OOD damping", LBLUE),
              ("confounded", "decorrelated", "confounded / decorr.", RED)]
    fig, ax = plt.subplots(figsize=(W*0.62, 1.75)); y = np.arange(len(scs))[::-1]; h = 0.26
    for j, (m, sh, lab, c) in enumerate(groups):
        mu, sd = ms([[r["monitor"][sh][k]["auroc_shift"] for k, _ in scs] for r in Rs[m]])
        ax.barh(y - (j-1)*h, mu, h, xerr=sd, color=c, label=lab, error_kw={"elinewidth": 0.6, "capsize": 1.2})
    ax.axvline(0.5, color="k", ls="--", lw=0.6); ax.set_xlim(0.35, 0.8); ax.set_yticks(y, [l for _, l in scs])
    ax.set_xlabel("AUROC (shifted vs. training distribution)"); ax.legend(loc="lower right", fontsize=6.2, handlelength=1)
    ax.grid(axis="y", visible=False)
    fig.savefig(f"{OUT}/fig_monitor.pdf", bbox_inches="tight"); plt.close(fig)

if __name__ == "__main__":
    fig_frames(); fig_main(); fig_planner(); fig_monitor(); print("figures written")
