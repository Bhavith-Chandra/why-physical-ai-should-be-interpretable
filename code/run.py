"""Full experimental harness. Usage: python run.py --seed S --mode {clean,confounded} --episodes N --epochs E
Writes results/<mode>_s<seed>.json. No number is hand-entered anywhere."""
import argparse, json, os, time, numpy as np, torch
from sklearn.linear_model import RidgeCV
from sklearn.metrics import roc_auc_score, r2_score
from scipy.stats import spearmanr
from wm import *
from kinematic_monitor import fit_angle_probe, kinematic_score

p = argparse.ArgumentParser()
p.add_argument("--seed", type=int, default=0); p.add_argument("--mode", default="clean")
p.add_argument("--episodes", type=int, default=1200); p.add_argument("--epochs", type=int, default=10)
a = p.parse_args()
torch.set_num_threads(2); os.makedirs("results", exist_ok=True)
log = lambda s: print(s, flush=True)
base = 1000 * a.seed + (0 if a.mode == "clean" else 500)   # disjoint RNG streams per run

t0 = time.time()
Xtr, Ytr, Str = dataset(a.episodes, a.mode, base + 1)
torch.manual_seed(a.seed)          # seed BEFORE construction: weight init must be reproducible
model = train(WorldModel(), Xtr, Ytr, epochs=a.epochs, seed=a.seed, log=log).eval()
log(f"trained in {time.time()-t0:.0f}s")

NT = 300
tests = {
    "val_matched":  dataset(NT, a.mode, base + 2),                       # same distribution as training
    "decorrelated": dataset(NT, "clean", base + 3),                      # colour independent of g
    "ood_gravity":  dataset(NT, "clean", base + 4, g_range=(16, 22)),
    "ood_damping":  dataset(NT, "clean", base + 5, damp_range=(0.6, 1.0)),
}

@torch.no_grad()
def run(X, bs=512):
    outs, zs = [], []
    for i in range(0, len(X), bs):
        o, z = model.rollout(X[i:i+bs], return_z=True); outs.append(o); zs.append(z)
    return torch.cat(outs), torch.cat(zs)

def ang_err(pred, S):
    a_hat = centroid_angle(pred).numpy()            # (N,H)
    return np.abs(np.angle(np.exp(1j*(a_hat - S[:, 5:5+H]))))

def implied_g(pred_angles, S):
    """Gravity the model *behaves as if* it believes: least-squares fit of th'' = -g sin th - b th' to its own rollout."""
    th0 = S[:, 0][:, None]; th = np.concatenate([th0, pred_angles], 1)
    w = np.diff(th, axis=1) / DT; acc = np.diff(w, axis=1) / DT
    s = np.sin(th[:, 1:-1]); wm_ = 0.5*(w[:, 1:] + w[:, :-1])
    gs = []
    for i in range(len(th)):
        A = np.stack([-s[i], -wm_[i]], 1); sol, *_ = np.linalg.lstsq(A, acc[i], rcond=None); gs.append(sol[0])
    return np.array(gs)

res = {"seed": a.seed, "mode": a.mode, "n_train_windows": len(Xtr), "epochs": a.epochs}
cache = {}
for name, (X, Y, S) in tests.items():
    pred, z = run(X); e = ang_err(pred, S)
    cache[name] = dict(X=X, Y=Y, S=S, pred=pred, z=z[:, 0].numpy(), err=e)
    res[name] = {"pixel_mse_x1e3": float(((pred - Y)**2).mean()*1e3),
                 "angle_err_h1": float(e[:, 0].mean()), "angle_err_hH": float(e[:, -1].mean()),
                 "angle_err_mean": float(e.mean())}
    log(f"{name}: {res[name]}")

# ---- E1: linear probes on the encoder latent z_t (fit on train split, score on held-out matched val)
_, ztr = run(Xtr); ztr = ztr[:, 0].numpy()
targets = {"sin_theta": lambda S: np.sin(S[:, 0]), "cos_theta": lambda S: np.cos(S[:, 0]),
           "omega": lambda S: S[:, 1], "gravity": lambda S: S[:, 2], "damping": lambda S: S[:, 3],
           "redness": lambda S: S[:, 4]}
probes, res["probe_r2"] = {}, {}
for k, f in targets.items():
    pr = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(ztr, f(Str)); probes[k] = pr
    res["probe_r2"][k] = {t: float(r2_score(f(cache[t]["S"]), pr.predict(cache[t]["z"])))
                          for t in ["val_matched", "decorrelated"]}
log(f"probe R2: {res['probe_r2']}")

# ---- E2: causal intervention on a nuisance variable (colour), physics held fixed
X, S = cache["decorrelated"]["X"], cache["decorrelated"]["S"]
Xf = X.view(len(X), K_IN, 3, R, R).flip(2).reshape_as(X)   # swap red<->blue channels = redness r -> 1-r
pf, zf = run(Xf)
a0 = centroid_angle(cache["decorrelated"]["pred"]).numpy(); a1 = centroid_angle(pf).numpy()
g0 = implied_g(a0, S); g1 = implied_g(a1, S)
red = S[:, 4]; dred = (1 - red) - red
res["intervention"] = {
    "mean_abs_angle_shift_hH": float(np.abs(np.angle(np.exp(1j*(a1[:, -1]-a0[:, -1])))).mean()),
    "implied_g_shift_per_unit_redness": float(np.polyfit(dred, g1 - g0, 1)[0]),
    "probe_g_shift_per_unit_redness": float(np.polyfit(dred, probes["gravity"].predict(zf[:, 0].numpy())
                                                       - probes["gravity"].predict(cache["decorrelated"]["z"]), 1)[0]),
    "implied_g_corr_true": float(np.corrcoef(g0, S[:, 2])[0, 1]),
    "implied_g_corr_redness": float(np.corrcoef(g0, red)[0, 1]),
}
log(f"intervention: {res['intervention']}")

# ---- E3: runtime monitoring. Scores computable BEFORE acting (from current input only) vs a lagged baseline.
rng = np.random.default_rng(base + 9)
ref = Xtr[rng.choice(len(Xtr), min(2000, len(Xtr)), replace=False)].flatten(1)
mu = ztr.mean(0); P = np.linalg.pinv(np.cov(ztr.T) + 1e-6*np.eye(ztr.shape[1]))
g_lo, g_hi = np.percentile(Str[:, 2], [1, 99])
def scores(name):
    c = cache[name]; zc = c["z"] - mu
    knn = torch.cdist(c["X"].flatten(1), ref).min(1).values.numpy()
    maha = np.einsum("ni,ij,nj->n", zc, P, zc)
    gp = probes["gravity"].predict(c["z"]); gprobe = np.maximum(g_lo - gp, 0) + np.maximum(gp - g_hi, 0)
    lag = ((c["pred"][:, 0] - c["Y"][:, 0])**2).flatten(1).mean(1).numpy()   # needs the next frame: arrives too late
    kin = kinematic_score(model, angle_probe, c["X"])
    return {"kinematic_consistency": kin, "pixel_knn": knn, "latent_mahalanobis": maha, "probe_physics_range": gprobe, "lagged_1step_residual": lag}
angle_probe = fit_angle_probe(model, Xtr, Str)
sid = scores("val_matched"); res["monitor"] = {}
for shift in ["ood_gravity", "ood_damping", "decorrelated"]:
    so = scores(shift); res["monitor"][shift] = {}
    errs = np.concatenate([cache["val_matched"]["err"].mean(1), cache[shift]["err"].mean(1)])
    for k in sid:
        s = np.concatenate([sid[k], so[k]]); y = np.r_[np.zeros(len(sid[k])), np.ones(len(so[k]))]
        res["monitor"][shift][k] = {"auroc_shift": float(roc_auc_score(y, s)),
                                    "spearman_vs_error": float(spearmanr(s, errs).correlation)}
log(f"monitor: {json.dumps(res['monitor'], indent=1)}")
out = "rerun" if os.environ.get("RERUN") else "results"; json.dump(res, open(f"{out}/{a.mode}_s{a.seed}.json", "w"), indent=1)
torch.save(model.state_dict(), f"{out}/{a.mode}_s{a.seed}.pt")
log(f"done {time.time()-t0:.0f}s")
