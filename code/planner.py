"""Tests Proposition 3 (planning is optimistic) with the trained world models.

A planner chooses one of N candidate situations (state + appearance) to act in, picking the one whose PREDICTED
cost is lowest. Cost = |theta at t+H| (task: be at the bottom of the swing at the decision horizon). We measure the
optimism of the chosen situation: realized cost - predicted cost. Prop. 3 predicts optimism >= 0 and growing with N.
A shortcut model is systematically wrong on color-physics mismatches, so the planner should find those.

Also: Monte Carlo check of the closed form E[optimism] = sigma * E[max of N iid N(0,1)] for unbiased Gaussian errors.
Writes results/planner.json."""
import json, glob, re, numpy as np, torch
from wm import *

torch.set_num_threads(2)
wrap = lambda x: np.angle(np.exp(1j*x))
NS = [1, 2, 4, 8, 16, 32, 64]; DRAWS = 3000

@torch.no_grad()
def final_angles(m, X, bs=512):
    return np.concatenate([centroid_angle(m.rollout(X[i:i+bs])).numpy()[:, -1] for i in range(0, len(X), bs)])

def plan(pred_cost, true_cost, rng):
    out = {}
    for n in NS:
        idx = rng.integers(0, len(pred_cost), (DRAWS, n))
        pick = idx[np.arange(DRAWS), pred_cost[idx].argmin(1)]
        out[n] = {"optimism": float((true_cost[pick] - pred_cost[pick]).mean()),
                  "realized": float(true_cost[pick].mean()), "predicted": float(pred_cost[pick].mean()),
                  "oracle_best": float(true_cost[idx].min(1).mean())}
    return out

seeds = sorted(int(re.search(r"clean_s(\d+)\.pt", f).group(1)) for f in glob.glob("results/clean_s*.pt"))
res = {"N": NS, "seeds": {}}
for s in seeds:
    Xd, _, Sd = dataset(300, "clean", 1000*s + 500 + 3)        # deployment: color independent of physics
    true = np.abs(Sd[:, 5 + H - 1])
    r = {}
    for mode in ["clean", "confounded"]:
        m = WorldModel(); m.load_state_dict(torch.load(f"results/{mode}_s{s}.pt")); m.eval()
        pred = np.abs(final_angles(m, Xd))
        r[mode] = plan(pred, true, np.random.default_rng(s))
    res["seeds"][s] = r; print(s, {k: [round(v[n]["optimism"], 3) for n in NS] for k, v in r.items()}, flush=True)

# Monte Carlo vs closed form (equal true costs, unbiased iid Gaussian errors, sigma=1)
rng = np.random.default_rng(0); mc = {}
for n in [1, 2, 4, 8, 16, 32, 64, 128, 256, 1024]:
    e = rng.standard_normal((20000, n)); opt = -e[np.arange(20000), e.argmin(1)]   # C - C_hat at argmin
    xs = np.linspace(-8, 8, 20001); phi = np.exp(-xs**2/2)/np.sqrt(2*np.pi)
    Phi = np.cumsum(phi)*(xs[1]-xs[0]); emax = float(np.trapezoid(xs * n * Phi**(n-1) * phi, xs))
    mc[n] = {"monte_carlo": float(opt.mean()), "exact_E_max": emax, "asymptote_sqrt_2logn": float(np.sqrt(2*np.log(n))) if n > 1 else 0.0}
res["gaussian_check"] = mc
json.dump(res, open("results/planner.json", "w"), indent=1)
print(json.dumps(mc, indent=0))
