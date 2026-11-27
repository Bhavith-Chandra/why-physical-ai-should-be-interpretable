"""Triangulation: check each headline claim with an independent method. Writes results/triangulation.json.
T1  angle readout     : centroid (paper) vs. argmax-pixel readout (independent) -> same error ordering?
T2  physics oracles   : simulator baselines. Does the causal model behave like 'knows state, not g'?
                        Does the confounded model on decorrelated data behave like 'g read from colour'?
T3  identifiability   : can g be read at all? nonlinear (kNN, MLP) probes on latents + oracle from true angles.
T4  dose-response     : graded colour interventions; shortcut effect should grow monotonically.
T5  uncertainty       : episode-level bootstrap 95% CIs for the headline gaps, per seed.
"""
import json, sys, numpy as np, torch
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
from wm import *

SEEDS = [int(s) for s in sys.argv[1:]] or [0, 1, 2]
torch.set_num_threads(2)
wrap = lambda x: np.angle(np.exp(1j*x))

def load(mode, s):
    m = WorldModel(); m.load_state_dict(torch.load(f"results/{mode}_s{s}.pt")); return m.eval()

@torch.no_grad()
def predict(m, X, bs=512):
    return torch.cat([m.rollout(X[i:i+bs]) for i in range(0, len(X), bs)])

@torch.no_grad()
def enc(m, X, bs=512):
    return torch.cat([m.enc(X[i:i+bs]) for i in range(0, len(X), bs)]).numpy()

def argmax_angle(fr):
    m = fr.clamp(0, 1).sum(-3).clone(); m[..., R//2-3:R//2+1, R//2-2:R//2+2] = 0
    idx = m.flatten(-2).argmax(-1); cy, cx = (idx // R).float(), (idx % R).float()
    return torch.atan2(cx - R/2, cy - (R/2 - 2)).numpy()

def roll_physics(S, g):
    """Simulate the true dynamics from the true current (theta, omega) with a chosen gravity; same integrator as data."""
    th, om, b = S[:, 0].copy(), S[:, 1].copy(), S[:, 3]
    out = []
    for _ in range(H):
        for _ in range(10):
            om = om + DT/10*(-g*np.sin(th) - b*om); th = th + DT/10*om
        out.append(th.copy())
    return np.stack(out, 1)

def boot_ci(x, n=2000, rng=np.random.default_rng(0)):
    bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]

out = {}
for s in SEEDS:
    r = {}
    bc, bk = 1000*s, 1000*s + 500
    mc, mk = load("clean", s), load("confounded", s)
    Xv_c, Yv_c, Sv_c = dataset(300, "clean", bc + 2)          # causal model's matched val
    Xv_k, Yv_k, Sv_k = dataset(300, "confounded", bk + 2)     # confounded model's matched val
    Xd, Yd, Sd = dataset(300, "clean", bk + 3)                # decorrelated set (as in run.py for confounded)

    # ---- T1: independent readout
    def errs(m, X, S):
        p = predict(m, X); a = argmax_angle(p); return np.abs(wrap(a - S[:, 5:5+H]))[:, -1]
    t1 = {"argmax_err_hH": {"clean_val": float(errs(mc, Xv_c, Sv_c).mean()),
                            "conf_val": float(errs(mk, Xv_k, Sv_k).mean()),
                            "conf_decorr": float(errs(mk, Xd, Sd).mean()),
                            "clean_decorr": float(errs(mc, Xd, Sd).mean())}}
    t1["ordering_holds"] = bool(t1["argmax_err_hH"]["conf_val"] < t1["argmax_err_hH"]["clean_val"] < t1["argmax_err_hH"]["conf_decorr"])
    r["T1_readout"] = t1

    # ---- T2: physics oracles on the decorrelated set (true trajectories S[:,5:])
    true = Sd[:, 5:5+H]
    oracle = lambda g: float(np.abs(wrap(roll_physics(Sd, g) - true))[:, -1].mean())
    r["T2_oracles_err_hH"] = {
        "true_g": oracle(Sd[:, 2]),
        "knows_state_g_fixed_at_mean_10": oracle(np.full(len(Sd), 10.0)),
        "g_read_from_colour_6+8r": oracle(6 + 8*Sd[:, 4]),
        "model_clean": float(np.abs(wrap(centroid_angle(predict(mc, Xd)).numpy() - true))[:, -1].mean()),
        "model_confounded": float(np.abs(wrap(centroid_angle(predict(mk, Xd)).numpy() - true))[:, -1].mean()),
    }

    # ---- T3: is gravity recoverable? nonlinear probes on the causal model's latent + physics ceiling
    Xt, _, St = dataset(1000, "clean", bc + 1)
    zt, zv = enc(mc, Xt), enc(mc, Xv_c)
    knn = KNeighborsRegressor(15).fit(zt, St[:, 2])
    mlp = MLPRegressor(hidden_layer_sizes=(128, 128), max_iter=400, random_state=s, early_stopping=True).fit(zt, St[:, 2])
    # physics ceiling: g from the 3 TRUE input angles (finite-difference acceleration), least squares with known damping
    def g_from_frames(X, S):
        th = centroid_angle(X.view(len(X), K_IN, 3, R, R)).numpy()
        w1, w2 = wrap(th[:, 1]-th[:, 0])/DT, wrap(th[:, 2]-th[:, 1])/DT
        acc = (w2 - w1)/DT; sn = np.sin(th[:, 1]); wm_ = 0.5*(w1+w2)
        return (-acc - S[:, 3]*wm_) / np.where(np.abs(sn) < 0.15, np.nan, sn)
    gf = g_from_frames(Xv_c, Sv_c); ok = np.isfinite(gf)
    # same estimator on EXACT simulator angles (perfect sensor) -> separates sensor limits from model limits
    rng2 = np.random.default_rng(bc + 11); th_, om_, g_, b_ = simulate(400, (6, 14), (0.05, 0.3), rng2)
    acc_ = (th_[:, 2]-2*th_[:, 1]+th_[:, 0])/DT**2; w_ = (th_[:, 2]-th_[:, 0])/(2*DT); sn_ = np.sin(th_[:, 1]); ok_ = np.abs(sn_) > 0.15
    ge_ = (-acc_ - b_*w_)/sn_
    rd = np.abs(wrap(centroid_angle(Xv_c.view(len(Xv_c), K_IN, 3, R, R)).numpy()[:, -1] - Sv_c[:, 0]))
    r["T3_gravity_identifiability"] = {
        "exact_angles_r2_(perfect_sensor)": float(r2_score(g_[ok_], ge_[ok_])),
        "centroid_readout_err_mean_rad": float(rd.mean()), "centroid_readout_err_max_rad": float(rd.max()),
        "knn_probe_r2": float(r2_score(Sv_c[:, 2], knn.predict(zv))),
        "mlp_probe_r2": float(r2_score(Sv_c[:, 2], mlp.predict(zv))),
        "physics_from_3_frames_r2_(|sin|>0.15)": float(r2_score(Sv_c[ok, 2], np.clip(gf[ok], 0, 30))),
        "physics_from_3_frames_frac_usable": float(ok.mean())}

    # ---- T4: dose-response of a graded colour intervention (shift redness by d, re-render from true states)
    rng = np.random.default_rng(bk + 77)
    th, om, g, b = simulate(300, (6, 14), (0.05, 0.3), rng)
    red0 = rng.uniform(0.2, 0.8, 300)
    def frames_for(red):
        col = np.stack([0.3+0.7*red, 0.3*np.ones_like(red), 0.3+0.7*(1-red)], 1)
        X, _, _ = windows(render(th, col), th, om, g, b, red); return X
    base = {k: centroid_angle(predict(m, frames_for(red0))).numpy()[:, -1] for k, m in [("clean", mc), ("conf", mk)]}
    dose = {"clean": [], "conf": []}
    for d in [0.05, 0.1, 0.2, 0.3]:
        Xs = frames_for(np.clip(red0 + d, 0, 1))
        for k, m in [("clean", mc), ("conf", mk)]:
            dose[k].append(float(np.abs(wrap(centroid_angle(predict(m, Xs)).numpy()[:, -1] - base[k])).mean()))
    r["T4_dose_response"] = {"deltas": [0.05, 0.1, 0.2, 0.3], **dose,
                             "conf_monotone": bool(np.all(np.diff(dose["conf"]) > 0))}

    # ---- T5: bootstrap CIs (episode-window level) for headline gaps
    ec = np.abs(wrap(centroid_angle(predict(mc, Xv_c)).numpy() - Sv_c[:, 5:5+H]))[:, -1]
    ek = np.abs(wrap(centroid_angle(predict(mk, Xv_k)).numpy() - Sv_k[:, 5:5+H]))[:, -1]
    ekd = np.abs(wrap(centroid_angle(predict(mk, Xd)).numpy() - Sd[:, 5:5+H]))[:, -1]
    r["T5_bootstrap95"] = {"clean_val": boot_ci(ec), "conf_val": boot_ci(ek), "conf_decorr": boot_ci(ekd),
                           "val_gap_excludes_0": bool(boot_ci(ec)[0] > boot_ci(ek)[1]),
                           "decorr_gap_excludes_0": bool(boot_ci(ekd)[0] > boot_ci(ek)[1])}
    out[f"s{s}"] = r; print(s, json.dumps(r, indent=1), flush=True)

import os; prev = json.load(open("results/triangulation.json")) if os.path.exists("results/triangulation.json") else {}; prev.update(out); json.dump(dict(sorted(prev.items())), open("results/triangulation.json", "w"), indent=1)
