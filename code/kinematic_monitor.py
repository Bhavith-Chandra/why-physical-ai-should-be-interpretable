"""Mechanistic pre-action monitor: disagreement between the acceleration the model imagines (decoded by linear
probes from its latent rollout z_0,z_1,z_2) and the acceleration visible in its own input frames."""
import numpy as np, torch
from sklearn.linear_model import RidgeCV
from wm import *

def wrap(x): return np.angle(np.exp(1j*x))

@torch.no_grad()
def latents(model, X, bs=512):
    return torch.cat([model.rollout(X[i:i+bs], h=2, return_z=True)[1] for i in range(0, len(X), bs)]).numpy()

def fit_angle_probe(model, Xtr, Str):
    z = latents(model, Xtr)[:, 0]
    ps = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(z, np.sin(Str[:, 0]))
    pc = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(z, np.cos(Str[:, 0]))
    return lambda zz: np.arctan2(ps.predict(zz), pc.predict(zz))

def kinematic_score(model, probe, X):
    Z = latents(model, X)                                   # (N, 3, z): z_t, z_{t+1}, z_{t+2}
    th_m = np.stack([probe(Z[:, k]) for k in range(3)], 1)
    a_model = wrap(th_m[:, 2] - th_m[:, 1]) - wrap(th_m[:, 1] - th_m[:, 0])        # imagined, centred at t+1
    fr = X.view(len(X), K_IN, 3, R, R)
    th_in = centroid_angle(fr).numpy()                       # observed input frames t-2..t
    a_obs = wrap(th_in[:, 2] - th_in[:, 1]) - wrap(th_in[:, 1] - th_in[:, 0])        # observed, centred at t-1
    # the two are two steps apart; compare magnitudes normalised by the restoring-force geometry
    return np.abs(a_model - a_obs) / DT**2
