"""Pixel pendulum world model: data, model, training. All numbers reported in the paper come from run.py outputs."""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

R = 32          # image resolution
DT = 0.08
K_IN = 3        # stacked input frames (acceleration observable)
H = 8           # prediction horizon (latent rollout)
T_EP = K_IN + H + 12

def simulate(n_ep, g_range, damp_range, rng):
    th = np.zeros((n_ep, T_EP)); om = np.zeros((n_ep, T_EP))
    g = rng.uniform(*g_range, n_ep); b = rng.uniform(*damp_range, n_ep)
    th[:, 0] = rng.uniform(-1.4, 1.4, n_ep); om[:, 0] = rng.uniform(-2, 2, n_ep)
    sub = 10
    for t in range(1, T_EP):
        a, w = th[:, t-1].copy(), om[:, t-1].copy()
        for _ in range(sub):  # semi-implicit Euler, fine substeps
            w = w + DT/sub * (-g*np.sin(a) - b*w)
            a = a + DT/sub * w
        th[:, t], om[:, t] = a, w
    return th, om, g, b

_yy, _xx = np.meshgrid(np.arange(R), np.arange(R), indexing="ij")

def render(th, color):
    """th: (N,T) angles; color: (N,3) RGB in [0,1]. Returns (N,T,3,R,R) float32."""
    cx = R/2 + 11*np.sin(th); cy = R/2 - 2 + 11*np.cos(th)
    d2 = (_xx[None, None] - cx[..., None, None])**2 + (_yy[None, None] - cy[..., None, None])**2
    blob = np.exp(-d2 / (2*1.6**2))
    img = blob[:, :, None] * color[:, None, :, None, None]
    # pivot marker (static, grey) so the frame has a fixed reference
    img[..., R//2-2:R//2, R//2-1:R//2+1] += 0.5
    return np.clip(img, 0, 1).astype(np.float32)

def make_colors(g, mode, rng, g_range=(6, 14)):
    n = len(g)
    if mode == "confounded":   # colour is a near-deterministic function of gravity
        r = np.clip((g - g_range[0])/(g_range[1]-g_range[0]) + rng.normal(0, 0.03, n), 0, 1)
    else:                      # colour independent of physics
        r = rng.uniform(0, 1, n)
    return np.stack([0.3 + 0.7*r, 0.3*np.ones(n), 0.3 + 0.7*(1-r)], 1), r

def windows(frames, th, om, g, b, redness):
    """Cut each episode into (K_IN input frames, H target frames) windows."""
    X, Y, S = [], [], []
    for s in range(0, T_EP - K_IN - H + 1, 3):
        X.append(frames[:, s:s+K_IN].reshape(len(g), -1, R, R))
        Y.append(frames[:, s+K_IN:s+K_IN+H])
        t = s + K_IN - 1
        S.append(np.stack([th[:, t], om[:, t], g, b, redness,
                           *[th[:, s+K_IN+k] for k in range(H)]], 1))
    return (torch.tensor(np.concatenate(X)), torch.tensor(np.concatenate(Y)),
            np.concatenate(S))
# S columns: 0 theta, 1 omega, 2 g, 3 damping, 4 redness, 5.. future thetas (H)

def dataset(n_ep, mode, seed, g_range=(6, 14), damp_range=(0.05, 0.3)):
    rng = np.random.default_rng(seed)
    th, om, g, b = simulate(n_ep, g_range, damp_range, rng)
    col, red = make_colors(g, mode, rng)
    return windows(render(th, col), th, om, g, b, red)

class WorldModel(nn.Module):
    def __init__(self, z=32):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv2d(3*K_IN, 32, 4, 2, 1), nn.GELU(),
                                 nn.Conv2d(32, 64, 4, 2, 1), nn.GELU(),
                                 nn.Conv2d(64, 64, 4, 2, 1), nn.GELU(),
                                 nn.Flatten(), nn.Linear(64*4*4, 256), nn.GELU(), nn.Linear(256, z))
        self.dyn = nn.Sequential(nn.Linear(z, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, z))
        self.dec_fc = nn.Sequential(nn.Linear(z, 256), nn.GELU(), nn.Linear(256, 64*4*4), nn.GELU())
        self.dec = nn.Sequential(nn.ConvTranspose2d(64, 64, 4, 2, 1), nn.GELU(),
                                 nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.GELU(),
                                 nn.ConvTranspose2d(32, 3, 4, 2, 1))
    def decode(self, z):
        return self.dec(self.dec_fc(z).view(-1, 64, 4, 4))
    def rollout(self, x, h=H, return_z=False):
        z = self.enc(x); zs, outs = [z], []
        for _ in range(h):
            z = z + self.dyn(z); zs.append(z); outs.append(self.decode(z))
        out = torch.stack(outs, 1)
        return (out, torch.stack(zs, 1)) if return_z else out

def train(model, X, Y, epochs=12, bs=128, lr=1e-3, seed=0, log=None):
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=epochs*((len(X)+bs-1)//bs))
    for ep in range(epochs):
        perm = torch.randperm(len(X)); tot = 0
        for i in range(0, len(X), bs):
            idx = perm[i:i+bs]
            y = Y[idx]; wgt = 1 + 30*y.amax(2, keepdim=True)  # up-weight ball pixels (ball ~1% of image)
            loss = (wgt*(model.rollout(X[idx]) - y)**2).mean() * 1000
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); tot += loss.item()*len(idx)
        if log: log(f"  epoch {ep+1}/{epochs} loss {tot/len(X):.4f}")
    return model

def centroid_angle(frames):
    """Recover pendulum angle from rendered/decoded frames via intensity centroid (pivot masked)."""
    m = frames.clamp(0, 1).sum(-3).clone()  # (..., R, R)
    m[..., R//2-3:R//2+1, R//2-2:R//2+2] = 0
    m = (m - 0.5*m.amax((-1, -2), keepdim=True)).clamp(min=0)  # keep only the blob core
    w = m.sum((-1, -2)) + 1e-8
    xx = torch.arange(R).float(); yy = torch.arange(R).float()
    cx = (m * xx).sum((-1, -2)) / w; cy = (m * yy[:, None]).sum((-1, -2)) / w
    return torch.atan2(cx - R/2, cy - (R/2 - 2))
