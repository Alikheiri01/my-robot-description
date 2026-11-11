"""
motion/models.py -- the networks (PyTorch).

!! NOT RUN YET: written without torch available. First run on your machine (from mvp_model/):  python3 tests/test_torch.py

MotionTransformer (variants full / hist / noocc / robot) and ConcatNet (variant concat, simple fusion without attention).

Two output styles (every variant exists in both):
    residual (default)  prediction = CVM(last 4 steps) + learned correction; starts exactly at CVM (last layer = 0)
    direct  (suffix _direct, e.g. full_direct)  prediction = the network output itself, no CVM added.
            The raw output is multiplied by DIRECT_SCALE (a fixed constant, 3 m) only so that a normal-sized
            network output can reach paths several metres long; it is NOT a motion model.

Input  hist (B,8,3)    pedestrian frame at the anchor, last row 0
       crop (B,80,80)  int8: -1 unknown, 0 free, 100 occupied
Output (B,12,2)        positions in metres at +0.5 ... +6.0 s
See docs/model-spec.md for the picture.
"""
import torch
import torch.nn as nn

N_HIST, N_FUT, DT = 8, 12, 0.5
DIRECT_SCALE = 3.0


def cvm_baseline(hist, k=4):
    v = (hist[:, -1, :2] - hist[:, -1 - k, :2]) / (DT * k)                   # (B,2)
    steps = DT * torch.arange(1, N_FUT + 1, device=hist.device, dtype=hist.dtype)
    return v[:, None, :] * steps[None, :, None]                              # (B,12,2)


def hist_features(hist):
    xy = hist[:, :, :2]
    d = torch.zeros_like(xy)
    d[:, 1:] = (xy[:, 1:] - xy[:, :-1]) / DT                                 # velocity, 0 for the first step
    yaw = hist[:, :, 2:3]
    return torch.cat([xy / 3.0, d, torch.cos(yaw), torch.sin(yaw)], dim=-1)  # (B,8,6)


def crop_onehot(crop, no_occ=False):
    c = crop.long()
    unk = (c == -1).float()
    occ = (c == 100).float()
    free = (c == 0).float()
    if no_occ:                                                               # occupied cells are shown as free
        free = free + occ
        occ = torch.zeros_like(occ)
    return torch.stack([unk, free, occ], dim=1)                              # (B,3,80,80)


def token_centres(n=10):
    """(n*n, 2) centre of every scene token in the pedestrian frame, scaled to about [-1, 1]; row-major (y rows, x cols)."""
    cell = 8.0 / n
    pts = [((-2.0 + cell * (j + 0.5)) / 4.0, (-4.0 + cell * (i + 0.5)) / 4.0) for i in range(n) for j in range(n)]
    return torch.tensor(pts, dtype=torch.float32)


class MotionTransformer(nn.Module):
    def __init__(self, d=64, heads=4, ff=128, n_hist_layers=2, n_cross_layers=2, n_scene_layers=1, drop=0.1,
                 use_crop=True, no_occ=False, use_robot=False, residual=True):
        super().__init__()
        self.use_crop, self.no_occ, self.use_robot, self.residual = use_crop, no_occ, use_robot, residual
        self.hist_in = nn.Linear(6, d)
        self.hist_pos = nn.Parameter(torch.zeros(1, N_HIST, d))
        enc = nn.TransformerEncoderLayer(d, heads, ff, drop, batch_first=True, norm_first=True, activation='gelu')
        self.hist_enc = nn.TransformerEncoder(enc, n_hist_layers, enable_nested_tensor=False)

        self.stem = nn.Sequential(                                           # 80 -> 40 -> 20 -> 10
            nn.Conv2d(3, 32, 3, 2, 1), nn.GELU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.GELU(),
            nn.Conv2d(64, d, 3, 2, 1))
        self.register_buffer('centres', token_centres(10))                  # position of a token = its (x, y) in metres
        self.pos_mlp = nn.Sequential(nn.Linear(2, d), nn.GELU(), nn.Linear(d, d))
        sc = nn.TransformerEncoderLayer(d, heads, ff, drop, batch_first=True, norm_first=True, activation='gelu')
        self.scene_enc = nn.TransformerEncoder(sc, n_scene_layers, enable_nested_tensor=False)
        self.robot_in = nn.Linear(2, d)

        dec = nn.TransformerDecoderLayer(d, heads, ff, drop, batch_first=True, norm_first=True, activation='gelu')
        self.cross = nn.TransformerDecoder(dec, n_cross_layers)              # queries = history tokens, memory = scene
        self.head = nn.Sequential(nn.LayerNorm(2 * d), nn.Linear(2 * d, 128), nn.GELU(), nn.Linear(128, N_FUT * 2))
        nn.init.normal_(self.hist_pos, std=0.02)
        if residual:
            nn.init.zeros_(self.head[-1].weight)                             # start exactly at CVM
            nn.init.zeros_(self.head[-1].bias)

    def forward(self, hist, crop, robot=None):
        B = hist.shape[0]
        h = self.hist_enc(self.hist_in(hist_features(hist)) + self.hist_pos)             # (B,8,d)
        if self.use_crop:
            s = self.stem(crop_onehot(crop, self.no_occ)).flatten(2).transpose(1, 2)     # (B,100,d)
            s = self.scene_enc(s + self.pos_mlp(self.centres)[None])
            if self.use_robot and robot is not None:
                s = torch.cat([s, self.robot_in(robot / 5.0)[:, None, :]], dim=1)
            mem = s
        else:
            mem = h                                                                      # history-only ablation
        z = self.cross(h, mem)                                                           # (B,8,d)
        feat = torch.cat([z[:, -1], z.mean(1)], dim=-1)
        out = self.head(feat).view(B, N_FUT, 2)
        return cvm_baseline(hist) + out if self.residual else DIRECT_SCALE * out


class ConcatNet(nn.Module):
    """Simple fusion, NO attention: GRU over the history + small CNN over the crop, vectors joined, MLP head."""

    def __init__(self, d=64, drop=0.1, residual=True):
        super().__init__()
        self.residual = residual
        self.gru = nn.GRU(6, d, batch_first=True)
        self.cnn = nn.Sequential(                                            # 80 -> 40 -> 20 -> 10 -> 5
            nn.Conv2d(3, 32, 3, 2, 1), nn.GELU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.GELU(),
            nn.Conv2d(64, 64, 3, 2, 1), nn.GELU(),
            nn.Conv2d(64, 64, 3, 2, 1), nn.GELU(),
            nn.Flatten(), nn.Linear(64 * 5 * 5, d), nn.GELU())
        self.head = nn.Sequential(nn.LayerNorm(2 * d), nn.Dropout(drop), nn.Linear(2 * d, 128), nn.GELU(),
                                  nn.Linear(128, N_FUT * 2))
        if residual:
            nn.init.zeros_(self.head[-1].weight)
            nn.init.zeros_(self.head[-1].bias)

    def forward(self, hist, crop, robot=None):
        _, hn = self.gru(hist_features(hist))                                # hn (1,B,d)
        feat = torch.cat([hn[0], self.cnn(crop_onehot(crop))], dim=-1)
        out = self.head(feat).view(hist.shape[0], N_FUT, 2)
        return cvm_baseline(hist) + out if self.residual else DIRECT_SCALE * out


_BASE = {
    'full': lambda **kw: MotionTransformer(**kw),
    'hist': lambda **kw: MotionTransformer(use_crop=False, **kw),
    'noocc': lambda **kw: MotionTransformer(no_occ=True, **kw),
    'robot': lambda **kw: MotionTransformer(use_robot=True, **kw),
    'concat': lambda **kw: ConcatNet(**kw),
}
VARIANTS = {}
for _name, _make in _BASE.items():
    VARIANTS[_name] = (lambda m=_make: m(residual=True))
    VARIANTS[_name + '_direct'] = (lambda m=_make: m(residual=False))

# "trained with damage" versions: the same networks, train.py damages the inputs while training (motion/damage_torch.py)
VARIANTS['full_dmg'] = VARIANTS['full']
VARIANTS['hist_dmg'] = VARIANTS['hist']


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def flip_batch(hist, crop, fut):
    """Mirror the world left/right (y -> -y). Must do exactly what motion.data.flip_arrays does (tests check this)."""
    h = hist.clone()
    h[..., 1] = -h[..., 1]
    h[..., 2] = -h[..., 2]
    f = fut.clone()
    f[..., 1] = -f[..., 1]
    if f.shape[-1] > 2:
        f[..., 2] = -f[..., 2]
    return h, torch.flip(crop, dims=[-2]), f