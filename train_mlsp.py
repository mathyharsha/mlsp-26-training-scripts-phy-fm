#!/usr/bin/env python3
"""
train_mlsp.py ── MLSP 2026 Ablation Study
Physics-Informed Flow Matching in a Low-Rank Factor Space.
MODIFICATION: Denoising setup. Trains against NOISY targets, evaluates against CLEAN targets.
"""

import argparse
import json
import math
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

# ── CLI ───────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="MLSP 2026 ablation – one configuration")
    p.add_argument("--r", type=int, default=6, help="Rank / number of latent modes")
    p.add_argument("--noise_level", type=float, default=0.0, help="Standard deviation of additive Gaussian noise")
    p.add_argument("--phys_weight", type=float, default=30.0, help="Weight on Helmholtz physics loss (0 = disabled)")
    p.add_argument("--fm_weight", type=float, default=1.0)
    p.add_argument("--recon_weight", type=float, default=20.0)
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--m", type=int, default=50, help="Spatial dimension")
    p.add_argument("--n", type=int, default=50, help="Temporal dimension")
    p.add_argument("--target_modes", type=int, default=3, help="Ground-truth number of sinusoidal modes in target")
    p.add_argument("--ode_steps_train", type=int, default=40, help="RK4 steps for training ODE")
    p.add_argument("--ode_steps_eval", type=int, default=40, help="RK4 steps for eval ODE")
    p.add_argument("--eval_batch", type=int, default=32, help="Batch size for eval reconstruction loss")
    p.add_argument("--save_interval", type=int, default=50, help="Checkpoint every N epochs")
    p.add_argument("--output_dir", type=str, default="results")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ── Data ─────────────────────────────────────────────────────────────
def generate_target(batch_size, m, n, device, num_modes, seed=None):
    """
    Generates the pure, CLEAN underlying wave dynamics.
    Noise injection is handled explicitly in main() for precise SNR control.
    """
    if seed is not None:
        torch.manual_seed(seed)
    t = torch.arange(m, device=device).float().view(1, m, 1) / m * 10.0
    x = torch.arange(n, device=device).float().view(1, 1, n) / n * 10.0
    t = t.expand(batch_size, m, n)
    x = x.expand(batch_size, m, n)
    M = torch.zeros(batch_size, m, n, device=device)
    for i in range(1, num_modes + 1):
        k, w = float(i), i * 0.5
        amp = torch.rand(batch_size, 1, 1, device=device) * 2.0 + 0.5
        M = M + amp * torch.sin(k * x + w * t)
    return M


# ── Model ─────────────────────────────────────────────────────────────
class SinusoidalEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, t):
        device = t.device
        h = self.dim // 2
        freq = math.log(10000) / (h - 1)
        freq = torch.exp(torch.arange(h, device=device) * -freq)
        emb = t * freq[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)

class Block1D(nn.Module):
    def __init__(self, in_c, out_c, t_dim):
        super().__init__()
        self.conv = nn.Conv1d(in_c, out_c, 3, padding=1)
        self.norm = nn.GroupNorm(8, out_c)
        self.act = nn.GELU()
        self.t_mlp = nn.Linear(t_dim, out_c)
    def forward(self, x, t_emb):
        h = self.norm(self.conv(x)) + self.t_mlp(t_emb).unsqueeze(-1)
        return self.act(h)

class UNet1DVelocityNet(nn.Module):
    def __init__(self, m, n, r, t_dim=64, base=64):
        super().__init__()
        self.m, self.n, self.r = m, n, r
        self.pad = math.ceil(max(m, n) / 4) * 4
        self.t_emb = nn.Sequential(
            SinusoidalEmbeddings(t_dim),
            nn.Linear(t_dim, t_dim * 2), nn.GELU(),
            nn.Linear(t_dim * 2, t_dim),
        )
        self.init_conv = nn.Conv1d(2 * r, base, 3, padding=1)
        self.d1 = Block1D(base, base * 2, t_dim); self.p1 = nn.MaxPool1d(2)
        self.d2 = Block1D(base * 2, base * 4, t_dim); self.p2 = nn.MaxPool1d(2)
        self.mid = Block1D(base * 4, base * 4, t_dim)
        self.u1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.uc1 = Block1D(base * 8, base * 2, t_dim)
        self.u2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.uc2 = Block1D(base * 4, base, t_dim)
        self.out = nn.Conv1d(base, 2 * r, 3, padding=1)

    def forward(self, t, z):
        if not isinstance(t, torch.Tensor):
            t = torch.as_tensor(t, dtype=z.dtype, device=z.device)
        t = t.view(-1)
        if t.shape[0] == 1:
            t = t.expand(z.size(0))
        t = t.unsqueeze(1)
        B = z.size(0)
        A = z[:, : self.m * self.r].view(B, self.m, self.r).permute(0, 2, 1)
        Bt = z[:, self.m * self.r :].view(B, self.n, self.r).permute(0, 2, 1)
        Ap = F.pad(A, (0, self.pad - self.m))
        Bp = F.pad(Bt, (0, self.pad - self.n))
        x = torch.cat([Ap, Bp], dim=1)
        te = self.t_emb(t)
        x0 = self.init_conv(x)
        x1 = self.d1(x0, te); x1p = self.p1(x1)
        x2 = self.d2(x1p, te); x2p = self.p2(x2)
        xm = self.mid(x2p, te)
        xu1 = self.uc1(torch.cat([self.u1(xm), x2], 1), te)
        xu2 = self.uc2(torch.cat([self.u2(xu1), x1], 1), te)
        out = self.out(xu2)
        Ao = out[:, : self.r, : self.m].permute(0, 2, 1).reshape(B, -1)
        Bo = out[:, self.r :, : self.n].permute(0, 2, 1).reshape(B, -1)
        return torch.cat([Ao, Bo], 1)


# ── ODE & Physics ────────────────────────────────────────────────────
def _rk4_step(v_net, z, t, dt):
    k1 = v_net(t, z)
    k2 = v_net(t + 0.5 * dt, z + 0.5 * dt * k1)
    k3 = v_net(t + 0.5 * dt, z + 0.5 * dt * k2)
    k4 = v_net(t + dt, z + dt * k3)
    return z + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

def ode_rk4(v_net, z0, num_steps=40):
    dt = torch.tensor(1.0 / num_steps, dtype=z0.dtype, device=z0.device)
    z = z0.clone()
    t = torch.tensor(0.0, dtype=z0.dtype, device=z0.device)
    for _ in range(num_steps):
        z = _rk4_step(v_net, z, t, dt)
        t = t + dt
    return z

def physics_residual(A, B, lam_A, lam_B):
    D2_A = A[:, :-2, :] - 2 * A[:, 1:-1, :] + A[:, 2:, :]
    res_A = D2_A + lam_A.view(1, 1, -1) * A[:, 1:-1, :]
    D2_B = B[:, :-2, :] - 2 * B[:, 1:-1, :] + B[:, 2:, :]
    res_B = D2_B + lam_B.view(1, 1, -1) * B[:, 1:-1, :]
    return (res_A ** 2).mean() + (res_B ** 2).mean()


# ── Helper functions ─────────────────────────────────────────────────
def rayleigh_eigenvalues(modes_np: np.ndarray) -> np.ndarray:
    L, r = modes_np.shape
    lams = np.zeros(r)
    if L < 3:
        return lams
    for k in range(r):
        x = modes_np[:, k]
        D2_x = x[:-2] - 2 * x[1:-1] + x[2:]
        xi = x[1:-1]
        denom = float(xi @ xi)
        lams[k] = float(-(xi @ D2_x) / denom) if denom > 1e-12 else 0.0
    return lams

def min_physics_loss_np(modes_np: np.ndarray) -> float:
    L, r = modes_np.shape
    if L < 3:
        return 0.0
    total = 0.0
    n_int = float(L - 2)
    for k in range(r):
        x = modes_np[:, k]
        D2_x = x[:-2] - 2 * x[1:-1] + x[2:]
        xi = x[1:-1]
        D2_sq = float(D2_x @ D2_x)
        xi_sq = float(xi @ xi)
        proj = float((xi @ D2_x) ** 2 / xi_sq) if xi_sq > 1e-12 else 0.0
        total += (D2_sq - proj) / n_int
    return total / r

def compute_mode_energies(A_unit_np: np.ndarray, B_np: np.ndarray):
    r = A_unit_np.shape[1]
    E = np.array([
        float(np.linalg.norm(A_unit_np[:, k]) ** 2 * np.linalg.norm(B_np[:, k]) ** 2)
        for k in range(r)
    ])
    return E, E / (E.sum() + 1e-12)


# ── Eval pass ────────────────────────────────────────────────────────
@torch.no_grad()
def eval_metrics(v_net, lambda_A, lambda_B, args, device, M_eval_batch_clean,
                 phys_seed: int = 9999, recon_seed: int = 8888):
    m, n, r = args.m, args.n, args.r

    # 1. Physics + mode metrics
    torch.manual_seed(phys_seed)
    z0_s = torch.randn(1, m * r + n * r, device=device)
    z1_s = ode_rk4(v_net, z0_s, num_steps=args.ode_steps_eval)
    A1_s = z1_s[:, : m * r].view(1, m, r)
    B1_s = z1_s[:, m * r :].view(1, n, r)
    norms_s = A1_s.norm(dim=1, keepdim=True) + 1e-8
    Au_s = A1_s / norms_s

    loss_phys_l = physics_residual(Au_s, B1_s, lambda_A, lambda_B).item()
    A_np = Au_s[0].cpu().numpy()
    B_np = B1_s[0].cpu().numpy()
    lam_A_ray = rayleigh_eigenvalues(A_np)
    lam_B_ray = rayleigh_eigenvalues(B_np)
    phys_min = min_physics_loss_np(A_np) + min_physics_loss_np(B_np)
    E_abs, E_frac = compute_mode_energies(A_np, B_np)

    # 2. EVALUATE AGAINST STRICTLY CLEAN TARGET
    torch.manual_seed(recon_seed)
    z0_b = torch.randn(args.eval_batch, m * r + n * r, device=device)
    z1_b = ode_rk4(v_net, z0_b, num_steps=args.ode_steps_eval)
    A1_b = z1_b[:, : m * r].view(args.eval_batch, m, r)
    B1_b = z1_b[:, m * r :].view(args.eval_batch, n, r)
    norms_b = A1_b.norm(dim=1, keepdim=True) + 1e-8
    Au_b = A1_b / norms_b
    M_gen = torch.bmm(Au_b, B1_b.transpose(1, 2))

    # Error is generated output vs Clean Physics Ground Truth
    loss_recon_clean = ((M_gen - M_eval_batch_clean) ** 2).mean().item()

    return {
        "loss_recon": loss_recon_clean,
        "loss_phys_learned": loss_phys_l,
        "loss_phys_min": phys_min,
        "rayleigh_A": lam_A_ray,
        "rayleigh_B": lam_B_ray,
        "lambda_A": lambda_A.detach().cpu().numpy(),
        "lambda_B": lambda_B.detach().cpu().numpy(),
        "mode_energy_abs": E_abs,
        "mode_energy_frac": E_frac,
    }


# ── Main ──────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    tag = f"r{args.r}_noise{args.noise_level:.2f}_phys{args.phys_weight:.1f}"
    out_dir = os.path.join(args.output_dir, tag)
    os.makedirs(out_dir, exist_ok=True)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"[{tag}] device={device}", flush=True)

    m, n, r = args.m, args.n, args.r
    v_net = UNet1DVelocityNet(m=m, n=n, r=r).to(device)

    # ── Target Manifold Setup (The Denoising Architecture) ───────────────
    
    # 1. Generate the pure underlying wave
    M_single_clean = generate_target(
        batch_size=1, m=m, n=n, device=device, 
        num_modes=args.target_modes, seed=args.seed
    )
    
    # 2. Standardize the clean ground-truth anchor (variance ~ 1.0)
    M_clean_norm = (M_single_clean - M_single_clean.mean()) / (M_single_clean.std() + 1e-6)

    # 3. Create the corrupted training observation
    # Since M_clean_norm has variance 1, noise_level directly scales the Signal-to-Noise Ratio
    if args.noise_level > 0.0:
        generator = torch.Generator(device=device).manual_seed(args.seed + 1)
        noise_tensor = torch.randn(M_clean_norm.size(), device=device, generator=generator)
        M_noisy = M_clean_norm + args.noise_level * noise_tensor
    else:
        M_noisy = M_clean_norm.clone()

    # 4. Expand to batches
    M_train_batch_noisy = M_noisy.expand(args.batch_size, m, n)      # Training sees NOISY
    M_eval_batch_clean = M_clean_norm.expand(args.eval_batch, m, n)  # Eval sees CLEAN
    # ─────────────────────────────────────────────────────────────────────

    lam_init = torch.tensor([0.001 * i ** 2 for i in range(1, r + 1)],
                            dtype=torch.float32, device=device)
    lambda_A = nn.Parameter(lam_init.clone())
    lambda_B = nn.Parameter(lam_init.clone())

    if args.phys_weight > 0.0:
        params = list(v_net.parameters()) + [lambda_A, lambda_B]
        print(f"[{tag}] λ_A, λ_B are TRAINABLE", flush=True)
    else:
        params = list(v_net.parameters())
        print(f"[{tag}] λ_A, λ_B are FROZEN", flush=True)

    optimizer = optim.Adam(params, lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    scalar_hist = {k: [] for k in [
        "loss_total", "loss_fm", "loss_recon_train", "loss_phys_train",
        "loss_recon_eval", "loss_phys_learned_eval", "loss_phys_min_eval",
    ]}
    arr_hist = {k: [] for k in [
        "rayleigh_A", "rayleigh_B", "lambda_A", "lambda_B",
        "mode_energy_abs", "mode_energy_frac",
    ]}
    save_epochs = []
    t_start = time.time()

    for epoch in range(args.epochs):
        v_net.train()
        optimizer.zero_grad()

        z0 = torch.randn(args.batch_size, m * r + n * r, device=device)
        z1 = ode_rk4(v_net, z0, num_steps=args.ode_steps_train)
        A1 = z1[:, : m * r].view(args.batch_size, m, r)
        B1 = z1[:, m * r :].view(args.batch_size, n, r)
        norms = A1.norm(dim=1, keepdim=True) + 1e-8
        A_unit = A1 / norms
        M_gen = torch.bmm(A_unit, B1.transpose(1, 2))

        # Training Reconstruction Loss: Computed against the NOISY target observation
        loss_recon = ((M_gen - M_train_batch_noisy) ** 2).mean()

        loss_phys = physics_residual(A_unit, B1, lambda_A, lambda_B)

        # Flow matching
        t_fm = torch.rand(args.batch_size, device=device) * 0.98 + 0.01
        z_t = (1 - t_fm.unsqueeze(1)) * z0 + t_fm.unsqueeze(1) * z1
        v_pred = v_net(t_fm, z_t)
        u_star = z1 - z0
        loss_fm = ((v_pred - u_star) ** 2).mean()

        loss = (args.fm_weight * loss_fm +
                args.recon_weight * loss_recon +
                args.phys_weight * loss_phys)

        loss.backward()
        optimizer.step()
        scheduler.step()

        is_last = (epoch == args.epochs - 1)
        if epoch % args.save_interval == 0 or is_last:
            v_net.eval()
            ev = eval_metrics(v_net, lambda_A, lambda_B, args, device, M_eval_batch_clean)

            scalar_hist["loss_total"].append(loss.item())
            scalar_hist["loss_fm"].append(loss_fm.item())
            scalar_hist["loss_recon_train"].append(loss_recon.item())
            scalar_hist["loss_phys_train"].append(loss_phys.item())
            scalar_hist["loss_recon_eval"].append(ev["loss_recon"])
            scalar_hist["loss_phys_learned_eval"].append(ev["loss_phys_learned"])
            scalar_hist["loss_phys_min_eval"].append(ev["loss_phys_min"])

            for k in arr_hist:
                arr_hist[k].append(ev[k])
            save_epochs.append(epoch)

            elapsed = time.time() - t_start
            lr_now = scheduler.get_last_lr()[0]
            print(f"[{tag}] ep {epoch:4d}/{args.epochs} | "
                  f"loss {loss.item():7.4f} | "
                  f"recon_train(NOISY) {loss_recon.item():.4f} | "
                  f"recon_eval(CLEAN) {ev['loss_recon']:.4f} | "
                  f"phys {loss_phys.item():.5f} | "
                  f"fm {loss_fm.item():.4f} | lr {lr_now:.2e} | {elapsed:5.0f}s",
                  flush=True)

    # ── Save ─────────────────────────────────────────────────────────
    ckpt_path = os.path.join(out_dir, "model.pt")
    torch.save({
        "v_net": v_net.state_dict(),
        "lambda_A": lambda_A.detach().cpu(),
        "lambda_B": lambda_B.detach().cpu(),
        "args": vars(args),
    }, ckpt_path)

    save_dict = {"epochs": np.array(save_epochs)}
    for k, v in scalar_hist.items():
        save_dict[k] = np.array(v, dtype=np.float32)
    for k, v in arr_hist.items():
        save_dict[k] = np.array(v, dtype=np.float32)

    np.savez(os.path.join(out_dir, "metrics.npz"), **save_dict)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    total_time = time.time() - t_start
    print(f"\n[{tag}] ✓ Done in {total_time/60:.1f} min | Outputs in {out_dir}")


if __name__ == "__main__":
    main()
