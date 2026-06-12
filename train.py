"""
Physics-Informed Conditional Flow Matching in a Low-Rank Factor Space.
IEEE MLSP 2026 — Main training script.

Usage:
    python train.py --w_fm 1.0 --w_recon 25.0 --w_phys 85.0 --w_cond 80.0 \
                    --run_name full_model --output_dir runs/
"""
import argparse
import json
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR


# ─────────────────────────────── CLI ────────────────────────────────────────
def get_args():
    p = argparse.ArgumentParser(
        description="Physics-Informed Conditional Flow Matching Training"
    )
    # Grid / rank
    p.add_argument("--m",             type=int,   default=50,    help="Spatial dimension")
    p.add_argument("--n",             type=int,   default=50,    help="Temporal dimension")
    p.add_argument("--r",             type=int,   default=6,     help="Low rank")
    # Data
    p.add_argument("--num_modes",     type=int,   default=3,     help="Wave modes in target")
    p.add_argument("--max_cond",      type=int,   default=50,    help="Max conditioning points")
    # Architecture
    p.add_argument("--base_channels", type=int,   default=64)
    p.add_argument("--cond_dim",      type=int,   default=24)
    p.add_argument("--time_emb_dim",  type=int,   default=64)
    # Optimisation
    p.add_argument("--epochs",        type=int,   default=2500)
    p.add_argument("--batch_size",    type=int,   default=16)
    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--num_steps",     type=int,   default=40,    help="ODE RK4 steps")
    # Loss weights
    p.add_argument("--w_fm",          type=float, default=1.0,   help="Flow matching weight")
    p.add_argument("--w_recon",       type=float, default=25.0,  help="Reconstruction weight")
    p.add_argument("--w_phys",        type=float, default=85.0,  help="Physics weight")
    p.add_argument("--w_cond",        type=float, default=80.0,  help="Conditioning weight")
    # Evaluation / logging
    p.add_argument("--eval_every",    type=int,   default=50,    help="Eval interval (epochs)")
    p.add_argument("--test_batches",  type=int,   default=4,
                   help="Test batches averaged per evaluation call")
    p.add_argument("--save_every",    type=int,   default=500,   help="Checkpoint interval")
    p.add_argument("--seed",          type=int,   default=42)
    # I/O
    p.add_argument("--run_name",      type=str,   default="run")
    p.add_argument("--output_dir",    type=str,   default="runs")
    return p.parse_args()


# ─────────────────────────── DATA GENERATION ────────────────────────────────
def generate_target(B, m, n, device, num_modes):
    t = torch.arange(n, device=device).float().view(1, 1, n).expand(B, m, n) / n * 10.0
    x = torch.arange(m, device=device).float().view(1, m, 1).expand(B, m, n) / m * 10.0
    M = torch.zeros(B, m, n, device=device)
    for i in range(1, num_modes + 1):
        amp = torch.rand(B, 1, 1, device=device) * 2.0 + 0.5
        M += amp * torch.sin(i * x + i * 0.5 * t)
    return M


def get_conditioning(M_norm, max_cond, device):
    """
    Sample sparse spatial observations from normalised field M_norm [B, m, n].

    Returns
    -------
    mask   : [B, m]          binary mask (1 = observed spatial row)
    values : [B, max_cond, n] observed time-series, zero-padded to max_cond rows
    """
    B, m, n = M_norm.shape
    mask   = torch.zeros(B, m,        device=device)
    values = torch.zeros(B, max_cond, n, device=device)
    for b in range(B):
        k        = torch.randint(5, max_cond + 1, (1,)).item()
        idx      = torch.randperm(m, device=device)[:k]
        mask[b, idx]     = 1.0
        values[b, :k, :] = M_norm[b, idx, :]
    return mask, values


# ───────────────────────────────── MODEL ────────────────────────────────────
class SinPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freq = torch.exp(
            torch.arange(half, device=t.device) * -(math.log(10000) / (half - 1))
        )
        if t.ndim == 1:
            t = t.unsqueeze(1)
        emb = t * freq[None]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class Block1D(nn.Module):
    def __init__(self, in_ch, out_ch, temb_dim):
        super().__init__()
        self.conv   = nn.Conv1d(in_ch, out_ch, 3, padding=1)
        self.norm   = nn.GroupNorm(8, out_ch)
        self.act    = nn.GELU()
        self.t_proj = nn.Linear(temb_dim, out_ch)

    def forward(self, x, t_emb):
        h = self.norm(self.conv(x)) + self.t_proj(t_emb).unsqueeze(-1)
        return self.act(h)


class VelocityNet(nn.Module):
    def __init__(self, m, n, r, max_cond, temb_dim=64, base_ch=64, cond_dim=24):
        super().__init__()
        self.m, self.n, self.r = m, n, r
        self.pad      = math.ceil(max(m, n) / 4) * 4
        self.cond_dim = cond_dim
        self.max_cond = max_cond

        self.t_mlp = nn.Sequential(
            SinPosEmb(temb_dim),
            nn.Linear(temb_dim, temb_dim * 2), nn.GELU(),
            nn.Linear(temb_dim * 2, temb_dim),
        )
        self.cond_enc = nn.Sequential(
            nn.Linear(max_cond * n, cond_dim * 4), nn.GELU(),
            nn.Linear(cond_dim * 4, cond_dim * 2), nn.GELU(),
            nn.Linear(cond_dim * 2, cond_dim),
        )
        self.in_conv = nn.Conv1d(2 * r + cond_dim, base_ch, 3, padding=1)

        self.d1, self.p1 = Block1D(base_ch,     base_ch * 2, temb_dim), nn.MaxPool1d(2)
        self.d2, self.p2 = Block1D(base_ch * 2, base_ch * 4, temb_dim), nn.MaxPool1d(2)
        self.mid         = Block1D(base_ch * 4, base_ch * 4, temb_dim)

        self.u1  = nn.Upsample(scale_factor=2, mode='nearest')
        self.uc1 = Block1D(base_ch * 8, base_ch * 2, temb_dim)
        self.u2  = nn.Upsample(scale_factor=2, mode='nearest')
        self.uc2 = Block1D(base_ch * 4, base_ch,     temb_dim)

        self.out_conv = nn.Conv1d(base_ch, 2 * r, 3, padding=1)

    def forward(self, t, z, cond_values=None, cond_mask=None):
        if not isinstance(t, torch.Tensor):
            t = torch.as_tensor(t, dtype=z.dtype, device=z.device)
        if t.ndim == 0:
            t = t.unsqueeze(0).expand(z.size(0))
        elif t.ndim == 1 and t.shape[0] == 1:
            t = t.expand(z.size(0))

        B = z.size(0)
        A_in  = z[:, :self.m * self.r].view(B, self.m, self.r).permute(0, 2, 1)
        B_in  = z[:, self.m * self.r:].view(B, self.n, self.r).permute(0, 2, 1)

        A_pad = F.pad(A_in, (0, self.pad - self.m))
        B_pad = F.pad(B_in, (0, self.pad - self.n))
        x = torch.cat([A_pad, B_pad], dim=1)

        if cond_values is not None and cond_mask is not None:
            ce  = self.cond_enc(cond_values.view(B, -1))
            ce  = ce * cond_mask.mean(dim=1, keepdim=True)
            x   = torch.cat([x, ce.unsqueeze(-1).expand(-1, -1, x.shape[-1])], dim=1)
        else:
            x = torch.cat([x, torch.zeros(B, self.cond_dim, x.shape[-1],
                                           dtype=z.dtype, device=z.device)], dim=1)

        te  = self.t_mlp(t)
        x0  = self.in_conv(x)
        x1  = self.d1(x0, te);  x1p = self.p1(x1)
        x2  = self.d2(x1p, te); x2p = self.p2(x2)
        xm  = self.mid(x2p, te)
        xu1 = self.uc1(torch.cat([self.u1(xm), x2], 1), te)
        xu2 = self.uc2(torch.cat([self.u2(xu1), x1], 1), te)
        out = self.out_conv(xu2)

        A_out = out[:, :self.r, :self.m].permute(0, 2, 1).reshape(B, -1)
        B_out = out[:, self.r:, :self.n].permute(0, 2, 1).reshape(B, -1)
        return torch.cat([A_out, B_out], dim=1)


# ────────────────────────────── ODE / RK4 ───────────────────────────────────
def ode_rk4(v_net, z0, cond_values=None, cond_mask=None, num_steps=40):
    """
    4th-order Runge–Kutta ODE integrator from t=0 to t=1.
    No @torch.no_grad decorator so gradients flow during training.
    """
    dt = torch.tensor(1.0 / num_steps, device=z0.device, dtype=z0.dtype)
    z  = z0.clone()
    t  = torch.zeros(z0.size(0), device=z0.device, dtype=z0.dtype)
    for _ in range(num_steps):
        k1 = v_net(t,            z,              cond_values, cond_mask)
        k2 = v_net(t + 0.5 * dt, z + 0.5*dt*k1,  cond_values, cond_mask)
        k3 = v_net(t + 0.5 * dt, z + 0.5*dt*k2,  cond_values, cond_mask)
        k4 = v_net(t + dt,        z +    dt*k3,   cond_values, cond_mask)
        z  = z + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        t  = t + dt
    return z


# ─────────────────────────────── LOSSES ─────────────────────────────────────
def physics_loss(A, B, lA, lB):
    """Helmholtz residual on spatial (A) and temporal (B) factor matrices."""
    def helm(U, lam):
        d = U[:, :-2, :] - 2 * U[:, 1:-1, :] + U[:, 2:, :]
        return (d + lam.view(1, 1, -1) * U[:, 1:-1, :]).pow(2).mean()
    return helm(A, lA) + helm(B, lB)


def test_physics_loss(A, B):
    """
    Helmholtz residual utilizing an optimal per-column eigenvalue selection.
    This assesses physical fidelity completely independently of trained lA and lB parameters.
    """
    def optimal_helm(U):
        d = U[:, :-2, :] - 2 * U[:, 1:-1, :] + U[:, 2:, :]  # Discrete Laplacian Lx
        U_inner = U[:, 1:-1, :]
        
        # Analytically find lambda minimizing ||d + lam * U_inner||^2 via least squares projection
        num = -(d * U_inner).sum(dim=1, keepdim=True)
        den = (U_inner * U_inner).sum(dim=1, keepdim=True) + 1e-8
        lam_opt = num / den
        
        residual = d + lam_opt * U_inner
        return residual.pow(2).mean()

    return optimal_helm(A) + optimal_helm(B)


def cond_loss_normalized(M_gen, M_ref, mask, n):
    """
    Conditioning MSE normalised by the actual number of conditioned
    (spatial × temporal) elements per sample.
    """
    sq_err  = (M_gen - M_ref).pow(2) * mask.unsqueeze(2)   # [B, m, n]
    n_elem  = mask.sum(dim=1) * n                            # [B]  true element count
    per_b   = sq_err.sum(dim=(1, 2)) / (n_elem + 1e-8)     # [B]
    return per_b.mean()


def fm_loss(v_net, z0, z1, t_samp, cond_values, cond_mask):
    """Conditional flow matching: velocity = z1 - z0."""
    z_t    = (1 - t_samp.unsqueeze(1)) * z0 + t_samp.unsqueeze(1) * z1
    v_pred = v_net(t_samp, z_t, cond_values, cond_mask)
    return F.mse_loss(v_pred, z1 - z0)


# ─────────────────────────────── EVALUATION ──────────────────────────────────
@torch.no_grad()
def evaluate(v_net, args, device, num_batches=4, base_seed=999):
    """
    Evaluate on freshly generated test data using fixed seeds for cross-ablation comparability.
    Preserves and restores the global training RNG state seamlessly.
    """
    v_net.eval()
    totals = dict(recon=0.0, cond=0.0, phys=0.0, fm=0.0)

    # Preserve training RNG state
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    for b in range(num_batches):
        torch.manual_seed(base_seed + b)

        B = args.batch_size
        M = generate_target(B, args.m, args.n, device, args.num_modes)
        mu  = M.mean(dim=(1, 2), keepdim=True)
        sig = M.std(dim=(1, 2),  keepdim=True) + 1e-6
        Mn  = (M - mu) / sig

        mask, vals = get_conditioning(Mn, args.max_cond, device)

        z0 = torch.randn(B, args.m * args.r + args.n * args.r, device=device)
        z1 = ode_rk4(v_net, z0, vals, mask, args.num_steps)

        A1  = z1[:, :args.m * args.r].view(B, args.m, args.r)
        B1  = z1[:, args.m * args.r:].view(B, args.n, args.r)
        Au  = A1 / (A1.norm(dim=1, keepdim=True) + 1e-8)
        Mg  = torch.bmm(Au, B1.transpose(1, 2))

        totals["recon"] += F.mse_loss(Mg, Mn).item()
        totals["cond"]  += cond_loss_normalized(Mg, Mn, mask, args.n).item()
        
        # Apply the analytical unregularized test physics evaluation
        totals["phys"]  += test_physics_loss(Au, B1 * sig).item()

        t_s = torch.rand(B, device=device) * 0.98 + 0.01
        totals["fm"] += fm_loss(v_net, z0, z1, t_s, vals, mask).item()

    # Restore training RNG state
    torch.set_rng_state(cpu_rng)
    if cuda_rng is not None:
        torch.cuda.set_rng_state_all(cuda_rng)

    v_net.train()
    return {f"test_{k}": v / num_batches for k, v in totals.items()}


# ────────────────────────────────── MAIN ────────────────────────────────────
def main():
    args = get_args()
    torch.manual_seed(args.seed)

    device = (torch.device("cuda") if torch.cuda.is_available() else
              torch.device("mps")  if torch.backends.mps.is_available() else
              torch.device("cpu"))
    print(f"Device: {device}")

    run_dir = os.path.join(args.output_dir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # ── Model ────────────────────────────────────────────────────────────────
    net = VelocityNet(
        m=args.m, n=args.n, r=args.r,
        max_cond=args.max_cond,
        temb_dim=args.time_emb_dim,
        base_ch=args.base_channels,
        cond_dim=args.cond_dim,
    ).to(device)

    lam0 = torch.tensor([0.001 * i**2 for i in range(1, args.r + 1)], device=device)
    lA   = nn.Parameter(lam0.clone())
    lB   = nn.Parameter(lam0.clone())

    opt   = optim.Adam(list(net.parameters()) + [lA, lB], lr=args.lr)
    sched = CosineAnnealingLR(opt, T_max=args.epochs)

    # ── Calculate True Discrete Target Eigenvalues ───────────────────────────
    dx = 10.0 / args.m
    dt = 10.0 / args.n
    
    # Grid discretization correction mappings: 2 * (1 - cos(k * d))
    exact_lA = [2 * (1 - math.cos(i * dx)) for i in range(1, args.num_modes + 1)]
    exact_lB = [2 * (1 - math.cos((i * 0.5) * dt)) for i in range(1, args.num_modes + 1)]
    
    print("\nTarget Discrete Eigenvalues (Accounting for Discretization Loss):")
    print(f"Spatial (lA) exact: {[round(x, 6) for x in exact_lA]}")
    print(f"Temporal (lB) exact: {[round(x, 6) for x in exact_lB]}\n")

    log = []
    t0  = time.time()

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(args.epochs):
        opt.zero_grad()

        M   = generate_target(args.batch_size, args.m, args.n, device, args.num_modes)
        mu  = M.mean(dim=(1, 2), keepdim=True)
        sig = M.std(dim=(1, 2),  keepdim=True) + 1e-6
        Mn  = (M - mu) / sig

        mask, vals = get_conditioning(Mn, args.max_cond, device)

        z0 = torch.randn(args.batch_size, args.m * args.r + args.n * args.r, device=device)
        z1 = ode_rk4(net, z0, vals, mask, args.num_steps)

        A1 = z1[:, :args.m * args.r].view(args.batch_size, args.m, args.r)
        B1 = z1[:, args.m * args.r:].view(args.batch_size, args.n, args.r)
        Au = A1 / (A1.norm(dim=1, keepdim=True) + 1e-8)
        Mg = torch.bmm(Au, B1.transpose(1, 2))

        l_recon = F.mse_loss(Mg, Mn)
        l_cond  = cond_loss_normalized(Mg, Mn, mask, args.n)
        l_phys  = physics_loss(Au, B1 * sig, lA, lB)
        t_fm    = torch.rand(args.batch_size, device=device) * 0.98 + 0.01
        l_fm    = fm_loss(net, z0, z1, t_fm, vals, mask)

        loss = (args.w_fm    * l_fm    +
                args.w_recon * l_recon +
                args.w_phys  * l_phys  +
                args.w_cond  * l_cond)

        loss.backward()
        opt.step()
        sched.step()

        if epoch % args.eval_every == 0:
            test = evaluate(net, args, device, num_batches=args.test_batches)
            record = {
                "epoch":       epoch,
                "train_total": loss.item(),
                "train_fm":    l_fm.item(),
                "train_recon": l_recon.item(),
                "train_cond":  l_cond.item(),
                "train_phys":  l_phys.item(),
                "lr":          sched.get_last_lr()[0],
                **test,
            }
            log.append(record)
            with open(os.path.join(run_dir, "metrics.json"), "w") as f:
                json.dump(log, f, indent=2)

            print(
                f"[{epoch:4d}/{args.epochs}] "
                f"total={loss.item():.4f} "
                f"fm={l_fm.item():.4f} "
                f"recon={l_recon.item():.4f} "
                f"cond={l_cond.item():.4f} "
                f"phys={l_phys.item():.4f} | "
                f"t_recon={test['test_recon']:.4f} "
                f"t_cond={test['test_cond']:.4f} "
                f"t_phys={test['test_phys']:.4f} "
                f"[{time.time() - t0:.0f}s]"
            )
            
            # Print eigenvalue tracking comparison
            learned_lA_sorted, _ = torch.sort(lA.data)
            learned_lB_sorted, _ = torch.sort(lB.data)
            print(f"    -> Target  lA: {[round(x, 5) for x in exact_lA]}")
            print(f"    -> Learned lA: {[round(x, 5) for x in learned_lA_sorted[:args.num_modes].tolist()]}")
            print(f"    -> Target  lB: {[round(x, 5) for x in exact_lB]}")
            print(f"    -> Learned lB: {[round(x, 5) for x in learned_lB_sorted[:args.num_modes].tolist()]}\n")

        if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            torch.save(
                {"epoch": epoch, "net": net.state_dict(),
                 "lA": lA.data, "lB": lB.data,
                 "opt": opt.state_dict(), "args": vars(args)},
                os.path.join(run_dir, f"ckpt_{epoch + 1}.pth"),
            )

    # ── Final checkpoint + high-sample-count test evaluation ─────────────────
    torch.save(
        {"net": net.state_dict(), "lA": lA.data, "lB": lB.data, "args": vars(args)},
        os.path.join(run_dir, "model_final.pth"),
    )

    final_test = evaluate(net, args, device, num_batches=16, base_seed=9999)
    with open(os.path.join(run_dir, "final_test_metrics.json"), "w") as f:
        json.dump(final_test, f, indent=2)

    print(f"\nTraining complete. Final test: {final_test}")


if __name__ == "__main__":
    main()
