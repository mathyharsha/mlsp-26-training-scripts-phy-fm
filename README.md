# Physics-Informed Flow Matching in a Low-Rank Factor Space

This repository contains training scripts for a physics-informed conditional flow matching (CFM) framework that generates wavefield data in a low-rank factor representation (M ≈ A Bᵗ), with discrete Helmholtz-type physics constraints enforced on the learned factors.

## Files

### `train.py`
Main training script for the full conditional flow matching model (IEEE MLSP 2026).

- Generates synthetic wavefield data as a superposition of sinusoidal modes.
- Learns a `VelocityNet` (UNet1D-style architecture) that defines the flow-matching velocity field, conditioned on sparse spatial observations of the wavefield.
- Integrates the learned velocity field with an RK4 ODE solver to map noise `z0` to factor-space samples `z1 = (A, B)`, which are recombined into the reconstructed wavefield `M_gen = A Bᵗ`.
- Total loss combines four terms:
  - **Flow matching loss** (`w_fm`): matches predicted velocity to `z1 - z0`.
  - **Reconstruction loss** (`w_recon`): MSE between generated and normalized target wavefield.
  - **Physics loss** (`w_phys`): Helmholtz residual on factor matrices A and B using learnable eigenvalues `lA`, `lB`.
  - **Conditioning loss** (`w_cond`): MSE on the observed (masked) spatial rows.
- Periodically evaluates on freshly generated test batches with fixed seeds, tracking reconstruction, conditioning, physics, and flow-matching losses, as well as how the learned eigenvalues compare to the analytically exact discrete eigenvalues.
- Saves config, per-epoch metrics, periodic checkpoints, and a final checkpoint + high-sample-count test evaluation.

### `train_mlsp.py`
Single-configuration ablation script for the MLSP 2026 study, focused on a **denoising** setup (no conditioning).

- Generates a single clean target wavefield (fixed seed) and, if `noise_level > 0`, creates a noisy version of it.
- Trains the `UNet1DVelocityNet` velocity field via the same RK4-integrated flow matching procedure as above, but:
  - **Training reconstruction loss** is computed against the **noisy** target.
  - **Evaluation reconstruction loss** is computed against the **clean** target — i.e., the model is assessed on its ability to denoise.
- Physics loss (Helmholtz residual with learnable `lambda_A`, `lambda_B`) and flow-matching loss are computed the same way as in `train.py`, but without any conditioning input.
- Includes helper routines for:
  - Rayleigh-quotient eigenvalue estimation from the learned factors.
  - Analytical minimum-possible physics loss for a given factor (for comparison against the learned-eigenvalue residual).
  - Mode-energy decomposition (absolute and fractional) of the learned low-rank factors.
- Saves a checkpoint, a `.npz` file with full training/eval metric histories (scalars and per-mode arrays), and a config JSON, all under `results/<tag>/` where `<tag>` encodes the rank, noise level, and physics weight.

## Common components

Both scripts share the same overall structure:
- A UNet-style 1D velocity network operating on concatenated, padded factor matrices `[A; B]`.
- RK4 integration of the learned ODE from `t=0` (noise) to `t=1` (data).
- A discrete Helmholtz physics residual on the spatial (A) and temporal (B) factors, with trainable per-mode eigenvalues.
- Cosine-annealed Adam optimization.

## Usage

```bash
# Full conditional model (train.py)
python train.py --w_fm 1.0 --w_recon 25.0 --w_phys 85.0 --w_cond 80.0 \
                 --run_name full_model --output_dir runs/

# Denoising ablation (train_mlsp.py)
python train_mlsp.py --r 6 --noise_level 0.2 --phys_weight 30.0 \
                      --output_dir results/
```
