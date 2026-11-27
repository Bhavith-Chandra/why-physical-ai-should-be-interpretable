"""Internal (latent-level) analyses on saved checkpoints. Regenerates the exact same datasets as run.py (same seeds)."""
import json, sys, numpy as np, torch
from sklearn.linear_model import RidgeCV
from wm import *

@torch.no_grad()
def rollout_from_z(model, z, h=H):
    outs = []
    for _ in range(h):
        z = z + model.dyn(z); outs.append(model.decode(z))
    return torch.stack(outs, 1)

def analyse(mode, seed, episodes=1000):
    base = 1000*seed + (0 if mode == "clean" else 500)
    m = WorldModel(); m.load_state_dict(torch.load(f"results/{mode}_s{seed}.pt")); m.eval()
    Xtr, _, Str = dataset(episodes, mode, base + 1)
    Xd, _, Sd = dataset(300, "clean", base + 3)
    with torch.no_grad():
        ztr = torch.cat([m.enc(Xtr[i:i+512]) for i in range(0, len(Xtr), 512)]).numpy()
        zd = m.enc(Xd)
    fit = lambda y: RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(ztr, y)
    p_red, p_g = fit(Str[:, 4]), fit(Str[:, 2])
    w_red, w_g = p_red.coef_, p_g.coef_
    cos = float(w_red @ w_g / (np.linalg.norm(w_red)*np.linalg.norm(w_g)))
    # Latent steering: move z along the colour direction by the amount that changes probed redness by +/-0.5,
    # leaving the input untouched; measure how much the *imagined motion* changes.
    u = torch.tensor(w_red / (w_red @ w_red), dtype=torch.float32)          # unit change in probed redness
    a0 = centroid_angle(rollout_from_z(m, zd)).numpy()
    shifts, gshift = [], []
    for d in (-0.5, 0.5):
        zs = zd + d*u
        a1 = centroid_angle(rollout_from_z(m, zs)).numpy()
        shifts.append(np.abs(np.angle(np.exp(1j*(a1 - a0))))[:, -1].mean())
        gshift.append(float((p_g.predict(zs.numpy()) - p_g.predict(zd.numpy())).mean() / d))
    # control: random direction of equal norm
    rng = np.random.default_rng(seed); r = rng.normal(size=u.shape); r = torch.tensor(r/np.linalg.norm(r)*u.norm().item()*0.5, dtype=torch.float32)
    ar = centroid_angle(rollout_from_z(m, zd + r)).numpy()
    return {"cos_gravity_vs_colour_probe": cos,
            "steer_colour_angle_shift_hH": float(np.mean(shifts)),
            "steer_random_angle_shift_hH": float(np.abs(np.angle(np.exp(1j*(ar - a0))))[:, -1].mean()),
            "steer_colour_probe_g_shift_per_unit_redness": float(np.mean(gshift))}

if __name__ == "__main__":
    out = {}
    import glob, os, re
    seeds = sorted(int(re.search(r"_s(\d+)\.pt", f).group(1)) for f in glob.glob("results/clean_s*.pt"))
    for s in seeds:
        for mode in ["clean", "confounded"]:
            out[f"{mode}_s{s}"] = analyse(mode, s); print(mode, s, out[f"{mode}_s{s}"], flush=True)
    json.dump(out, open("results/posthoc.json", "w"), indent=1)
