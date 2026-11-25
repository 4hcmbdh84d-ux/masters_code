import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Save plots without opening Mac windows overnight.
import matplotlib.pyplot as plt
import os
import time
import csv
from pathlib import Path
from scipy.spatial import cKDTree
import json

# Reproducible independent experiment. Does not modify original baseline results.
SEED = int(os.environ.get("HEX_SEED", "101"))
SMOKE = os.environ.get("HEX_SMOKE", "0") == "1"
SMOKE_TAG = os.environ.get("HEX_SMOKE_TAG", "manual")
torch.manual_seed(SEED)
np.random.seed(SEED)

# ============================================================
# Device setup
# ============================================================
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")

# ============================================================
# Parameters
# ============================================================
N_points = 128 if SMOKE else 10000
#N_points = 17000
dim = 2
box_size = 120.0 if SMOKE else 1000.0
#box_size = 1300.0
latent_dim = 16
n_k = 20 if SMOKE else 120
epochs = int(os.environ.get("HEX_EPOCHS", "3" if SMOKE else "15000"))
lr = 1e-4
batch_size = 1
S_target = 0.0001
noise = 0.005
cutoff = 25.0
snapshot_every = 2 if SMOKE else 100

# Memory controls: exact repulsion, processed in blocks.
REPULSION_BLOCK_SIZE = 32 if SMOKE else 256
MEMORY_REPORT_EVERY = 100

# Max per-particle displacement as a fraction of box_size (was hardcoded
# inside PeriodicGenerator.forward(); see note at top of file).
scale_factor = 0.05

# Relative weights of the three loss contributions (Sec. III of the paper).
# Defaults reproduce the paper's unweighted sum L = L_hyper + L_rep + L_smooth.
lambda_hyper = 1.0
lambda_rep = 1.0
lambda_smooth = 1.0

# The ONLY additional weighted objective. Original 3 terms remain identical.
lambda_hex = float(os.environ.get("HEX_LAMBDA", "0.005"))
hex_alignment_weight = float(os.environ.get("HEX_BETA", "0.25"))
if lambda_hex < 0 or hex_alignment_weight < 0:
    raise ValueError("Hexatic objective weights must be nonnegative")

# Create snapshots folder
RUN_DIR = str(Path(__file__).resolve().parent / "hexatic_experiments" / (
    f"_smoke_{SMOKE_TAG}_seed_{SEED}_lambda_{lambda_hex:g}_beta_{hex_alignment_weight:g}" if SMOKE else
    f"seed_{SEED}_lambda_{lambda_hex:g}_beta_{hex_alignment_weight:g}"))
os.makedirs(RUN_DIR, exist_ok=True)
os.makedirs(os.path.join(RUN_DIR, "snapshots"), exist_ok=True)
CHECKPOINT_PATH = os.path.join(RUN_DIR, "checkpoint_latest.pt")
CHECKPOINT_EVERY = 2 if SMOKE else 500
LOG_PATH = os.path.join(RUN_DIR, "loss_history.csv")
start_time = time.perf_counter()
with open(os.path.join(RUN_DIR, "experiment_settings.json"), "w") as sf:
    json.dump({"seed": SEED, "smoke": SMOKE, "N_points": N_points,
               "box_size": box_size, "n_k": n_k, "epochs_requested": epochs,
               "lambda_hex": lambda_hex, "hex_alignment_weight": hex_alignment_weight,
               "repulsion_block_size": REPULSION_BLOCK_SIZE,
               "nearest_neighbours": 6, "nearest_method": "periodic scipy cKDTree",
               "notes": "Original three loss definitions and weights unchanged"}, sf, indent=2)

def sync_device():
    if device.type == "mps":
        torch.mps.synchronize()

def save_checkpoint(next_epoch):
    sync_device()
    data = {
        "next_epoch": next_epoch,
        "model": G.state_dict(),
        "optimizer": optimizer.state_dict(),
        "torch_rng": torch.get_rng_state(),
        "numpy_rng": np.random.get_state(),
        "elapsed_seconds": elapsed_before + time.perf_counter() - start_time,
        "parameters": {"N_points": N_points, "dim": dim, "box_size": box_size,
                       "latent_dim": latent_dim, "n_k": n_k, "lr": lr,
                       "batch_size": batch_size, "S_target": S_target,
                       "noise": noise, "cutoff": cutoff, "scale_factor": scale_factor,
                       "lambda_hyper": lambda_hyper, "lambda_rep": lambda_rep,
                       "lambda_smooth": lambda_smooth,
                       "lambda_hex": lambda_hex, "hex_alignment_weight": hex_alignment_weight,
                       "seed": SEED},
    }
    if device.type == "mps":
        data["mps_rng"] = torch.mps.get_rng_state()
    tmp = CHECKPOINT_PATH + ".tmp"
    torch.save(data, tmp)
    os.replace(tmp, CHECKPOINT_PATH)
    print(f"Checkpoint saved after epoch {next_epoch}", flush=True)

# ============================================================
# Minimum image displacement
# ============================================================
def mic_displacement(d, box_size):
    return d - torch.round(d / box_size) * box_size

# ============================================================
# Memory-efficient EXACT repulsion with a custom backward pass.
# The old implementation materialised an N x N x 2 displacement tensor.
# This computes every ordered (i,j) pair, but only BLOCK_SIZE rows at once.
# The mask and denominator are identical to the original definition.
# No approximation, subsampling, or change to the three loss weights.
# ============================================================
class BlockwiseRepulsion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, coords, box_size, cutoff, block_size):
        n = coords.shape[0]
        eps = 1e-6
        total = coords.new_zeros(())
        count = coords.new_zeros((), dtype=torch.long)
        cutoff2 = cutoff * cutoff
        with torch.no_grad():
            for lo in range(0, n, block_size):
                hi = min(lo + block_size, n)
                diff = coords[None, :, :] - coords[lo:hi, None, :]
                diff = mic_displacement(diff, box_size)
                dist2 = (diff * diff).sum(dim=-1) + eps
                mask = dist2 < cutoff2
                row_ids = torch.arange(lo, hi, device=coords.device)
                mask[torch.arange(hi-lo, device=coords.device), row_ids] = False
                total += torch.where(mask, dist2.reciprocal(), 0.0).sum()
                count += mask.sum()
        ctx.save_for_backward(coords)
        ctx.box_size, ctx.cutoff2, ctx.block_size = box_size, cutoff2, block_size
        ctx.count = count
        return total / count.clamp(min=1)

    @staticmethod
    def backward(ctx, grad_output):
        (coords,) = ctx.saved_tensors
        n = coords.shape[0]
        grad_coords = torch.zeros_like(coords)
        count = ctx.count
        # The cutoff mask and pair count are piecewise constant, as in the
        # original torch.where implementation. At cutoff boundaries the
        # original also does not differentiate through the comparison.
        with torch.no_grad():
            for lo in range(0, n, ctx.block_size):
                hi = min(lo + ctx.block_size, n)
                diff = coords[None, :, :] - coords[lo:hi, None, :]
                diff = mic_displacement(diff, ctx.box_size)
                dist2 = (diff * diff).sum(dim=-1) + 1e-6
                mask = dist2 < ctx.cutoff2
                row_ids = torch.arange(lo, hi, device=coords.device)
                mask[torch.arange(hi-lo, device=coords.device), row_ids] = False
                # diff = coords[j] - coords[i]. For 1/dist2, derivative
                # w.r.t. coords[i] is +2*diff/dist2**2; coords[j] is negative.
                factor = torch.where(mask, 2.0 / (dist2 * dist2), 0.0)
                pair_grad = diff * factor[..., None]
                grad_coords[lo:hi] += pair_grad.sum(dim=1)
                grad_coords -= pair_grad.sum(dim=0)
        grad_coords *= grad_output / count.clamp(min=1)
        return grad_coords, None, None, None



# ============================================================
# NEW: Sixfold local orientational ordering (hexatic objective)
# ============================================================
def periodic_six_neighbours(coords, box_length):
    """Return (N,6) nearest-neighbour indices with periodic boundaries.

    The neighbour search is intentionally non-differentiable: selecting a
    neighbour is discrete. Gradients flow through the subsequently computed
    bond vectors and psi6 values, NOT through the neighbour indices.

    KDTree needs only a small CPU copy of the N x 2 positions and avoids
    creating an N x N tensor on a memory-limited MacBook.
    """
    if coords.shape[0] < 7:
        raise ValueError("Need at least seven particles for six neighbours")
    x = np.mod(coords.detach().cpu().numpy().astype(np.float64), box_length)
    tree = cKDTree(x, boxsize=box_length)
    indices = tree.query(x, k=7, workers=1)[1]
    # Remove each particle itself explicitly (rather than assuming column 0).
    mask = indices != np.arange(len(x))[:, None]
    if not np.all(mask.sum(axis=1) == 6):
        raise RuntimeError("Unexpected nearest-neighbour query, check duplicates")
    indices = indices[mask].reshape(len(x), 6)
    return torch.as_tensor(indices.copy(), device=coords.device, dtype=torch.long)


def hexatic_order_loss(coords, box_length, alignment_weight=0.25):
    """Return L_hex, L_local, L_align, psi6 real/imag and mean |psi6|.

    psi6(i) = mean_{six nearest j} exp(6i*theta_ij).
    L_local = mean(1-|psi6|^2)
    L_align = mean_{directed i->j} |psi6(i)-psi6(j)|^2
    L_hex = L_local + alignment_weight * L_align.
    """
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("Hexatic loss requires a (N,2) coordinate array")
    neighbors = periodic_six_neighbours(coords, box_length)
    bond = mic_displacement(coords[neighbors] - coords[:, None, :], box_length)
    # Define a finite orientation in the pathological exact-coincidence case.
    bond2 = bond.square().sum(-1)
    safe_dx = torch.where(bond2 < 1e-18, torch.ones_like(bond[..., 0]), bond[..., 0])
    safe_dy = torch.where(bond2 < 1e-18, torch.zeros_like(bond[..., 1]), bond[..., 1])
    theta = torch.atan2(safe_dy, safe_dx)
    psi_re = torch.cos(6.0 * theta).mean(dim=1)
    psi_im = torch.sin(6.0 * theta).mean(dim=1)
    magnitude_squared = psi_re.square() + psi_im.square()
    local = (1.0 - magnitude_squared).mean()
    align = ((psi_re[:, None] - psi_re[neighbors]).square() +
             (psi_im[:, None] - psi_im[neighbors]).square()).mean()
    total = local + alignment_weight * align
    return total, local, align, psi_re, psi_im, magnitude_squared.clamp_min(0).sqrt()

def report_memory(epoch):
    if device.type == "mps":
        gb = 1024**3
        allocated = torch.mps.current_allocated_memory() / gb
        driver = torch.mps.driver_allocated_memory() / gb
        recommended = torch.mps.recommended_max_memory() / gb
        print(f"[memory epoch {epoch}] MPS tensors={allocated:.2f} GiB; "
              f"MPS driver={driver:.2f} GiB; recommended max={recommended:.2f} GiB",
              flush=True)

# ============================================================
# Generator network
# ============================================================
class PeriodicGenerator(nn.Module):
    def __init__(self, latent_dim, N_points, dim, box_size=1.0, scale_factor=0.05):
        super().__init__()
        self.N_points = N_points
        self.dim = dim
        self.box_size = box_size
        self.scale_factor = scale_factor

        # Fixed base positions
        base = torch.rand(1, N_points, dim) * box_size
        self.register_buffer("base", base)

        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, N_points * dim)
        )

    def forward(self, z):
        batch = z.size(0)
        base = self.base.expand(batch, -1, -1)
        delta_raw = self.net(z).view(batch, self.N_points, self.dim)
        delta = self.scale_factor * self.box_size * torch.tanh(delta_raw)
        coords = (base + delta) % self.box_size
        return coords

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0.0, 0.05)
        nn.init.constant_(m.bias, 0.0)

G = PeriodicGenerator(latent_dim, N_points, dim, box_size=box_size, scale_factor=scale_factor).to(device)
G.apply(init_weights)
optimizer = optim.Adam(G.parameters(), lr=lr)

# ============================================================
# k-vectors for S(k)
# ============================================================
freqs = [torch.fft.fftfreq(n_k, d=1.0/n_k) * 2*np.pi/box_size for _ in range(dim)]
mesh = torch.meshgrid(*freqs, indexing="ij")
kvecs = torch.stack([m.flatten() for m in mesh], dim=1).to(device)
mask_nonzero = ~(torch.all(kvecs == 0, dim=1))
kvecs = kvecs[mask_nonzero]
k_mag = torch.sqrt((kvecs**2).sum(dim=1))
mask_smallk = k_mag < (2*np.pi*4/box_size)

# Report the constrained-mode fraction chi = m/(d*N) referenced in the paper
# (Introduction / Conclusions). mask_smallk counts +k and -k separately,
# but S(k)=S(-k) makes them redundant, so the number of independent
# constrained modes is half the raw count.
m_raw = int(mask_smallk.sum().item())
m_independent = m_raw // 2
chi = m_independent / (dim * N_points)
print(f"Constrained modes: {m_raw} raw (+/-k pairs), {m_independent} independent")
print(f"chi = m/(d*N) = {chi:.6f}")

# ============================================================
# Structure factor (NumPy)
# ============================================================
def structure_factor_np(positions, n_k=40, box_size=1.0):
    dim = positions.shape[1]
    freqs = [np.fft.fftfreq(n_k, d=1.0/n_k) * 2*np.pi/box_size for _ in range(dim)]
    mesh = np.meshgrid(*freqs, indexing="ij")
    kvecs = np.stack([m.flatten() for m in mesh], axis=1)
    mask_nonzero = ~(np.all(kvecs == 0, axis=1))
    kvecs = kvecs[mask_nonzero]
    k_mag = np.sqrt((kvecs**2).sum(axis=1))
    phase = np.exp(-1j * positions @ kvecs.T)
    rho_k = phase.sum(axis=0)
    S_k = (np.abs(rho_k)**2 / len(positions)).real
    return k_mag, S_k

def radial_average(k_mag, S_k, bins=60):
    k_bins = np.linspace(0, k_mag.max(), bins)
    S_rad = np.zeros(len(k_bins)-1)
    for i in range(len(k_bins)-1):
        mask = (k_mag >= k_bins[i]) & (k_mag < k_bins[i+1])
        if np.any(mask):
            S_rad[i] = S_k[mask].mean()
    k_centers = 0.5*(k_bins[1:] + k_bins[:-1])
    return k_centers, S_rad

# ============================================================
# Training loop (vectorized)
# ============================================================
# Resume automatically if a checkpoint exists.
start_epoch = 0
elapsed_before = 0.0
if os.path.exists(CHECKPOINT_PATH):
    # Only resume from a checkpoint you created/trust.
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    expected = ckpt["parameters"]
    for key in ("N_points", "dim", "box_size", "latent_dim", "n_k", "lr",
                "batch_size", "S_target", "noise", "cutoff", "scale_factor",
                "lambda_hyper", "lambda_rep", "lambda_smooth",
                "lambda_hex", "hex_alignment_weight", "seed"):
        current_value = SEED if key == "seed" else globals()[key]
        if current_value != expected[key]:
            raise ValueError(f"Parameter {key} changed since checkpoint; refusing to resume.")
    G.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    torch.set_rng_state(ckpt["torch_rng"].detach().to(device="cpu", dtype=torch.uint8))
    np.random.set_state(ckpt["numpy_rng"])
    if device.type == "mps" and "mps_rng" in ckpt:
        torch.mps.set_rng_state(ckpt["mps_rng"].detach().to(device="cpu", dtype=torch.uint8))
    start_epoch = ckpt["next_epoch"]
    elapsed_before = ckpt["elapsed_seconds"]
    print(f"Resuming from epoch {start_epoch}", flush=True)
else:
    with torch.no_grad():
        np.savetxt(os.path.join(RUN_DIR, "initial_base_positions.csv"),
                   G.base[0].cpu().numpy(), delimiter=",", header="x,y", comments="")

# Save actual fixed base positions AFTER loading a checkpoint, so resume does
# not accidentally overwrite the Poisson reference with a fresh random base.
with torch.no_grad():
    np.savetxt(os.path.join(RUN_DIR, "points_poisson_reference.csv"),
               G.base[0].cpu().numpy(), delimiter=",", header="x,y", comments="")

# Resume hygiene: a checkpoint saved after epoch 8000 represents the state
# BEFORE the next iteration (zero-based index 8000). Any CSV rows beyond
# 8000 and any snapshots labelled 08000 or later belong to uncheckpointed
# work and would otherwise be duplicated on resume.
if start_epoch > 0:
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, newline="") as f:
            old_rows = list(csv.reader(f))
        if old_rows:
            kept_rows = [old_rows[0]] + [
                row for row in old_rows[1:]
                if row and int(row[0]) <= start_epoch
            ]
            if len(kept_rows) < len(old_rows):
                backup = LOG_PATH + ".before_resume.bak"
                if not os.path.exists(backup):
                    import shutil
                    shutil.copy2(LOG_PATH, backup)
                with open(LOG_PATH, "w", newline="") as f:
                    csv.writer(f).writerows(kept_rows)
                print(f"Removed {len(old_rows)-len(kept_rows)} uncheckpointed log rows; "
                      f"backup: {backup}", flush=True)
    for path in Path(RUN_DIR, "snapshots").glob("*_epoch_*.*"):
        try:
            snapshot_index = int(path.stem.rsplit("_epoch_", 1)[1])
        except (ValueError, IndexError):
            continue
        if snapshot_index >= start_epoch:
            path.unlink()
            print(f"Removed uncheckpointed snapshot: {path}", flush=True)

log_exists = os.path.exists(LOG_PATH)
logfile = open(LOG_PATH, "a", newline="", buffering=1)
writer = csv.writer(logfile)
if not log_exists:
    writer.writerow(["epoch", "total", "hyper", "repulsion", "smooth",
                     "hex", "hex_local", "hex_align", "mean_abs_psi6",
                     "global_abs_psi6", "Cx_unwrapped", "Cy_unwrapped", "elapsed_seconds"])

try:
 for epoch in range(start_epoch, epochs):
     z = torch.randn(batch_size, latent_dim, device=device)
     coords = G(z)[0] + noise * torch.randn(N_points, dim, device=device)
     coords = coords % box_size

     # Structure factor
     phase = torch.exp(-1j * (coords @ kvecs.T))
     rho_k = phase.sum(dim=0)
     S_k = (rho_k.abs()**2 / N_points).real
     L_hyper = ((S_k[mask_smallk] - S_target)**2).mean()

     # Exact same conditional average over all qualifying ordered pairs.
     # Custom backward recomputes one block at a time, keeping peak memory low.
     L_repulsion = BlockwiseRepulsion.apply(
         coords, box_size, cutoff, REPULSION_BLOCK_SIZE
     )

     # Smoothness term
     L_smooth = ((S_k - S_k.mean())**2).mean()

     # NEW fourth loss: six-neighbour local bond order + local alignment.
     # The existing three loss formulas and their unit weights are unchanged.
     L_hex, L_hex_local, L_hex_align, psi_re, psi_im, psi_abs = hexatic_order_loss(
         coords, box_size, hex_alignment_weight)
     loss = (lambda_hyper * L_hyper + lambda_rep * L_repulsion +
             lambda_smooth * L_smooth + lambda_hex * L_hex)

     optimizer.zero_grad(set_to_none=True)
     loss.backward()
     torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=1.0)
     optimizer.step()
     sync_device()
     with torch.no_grad():
         # Periodic-unwrapped positional Pearson diagnostic; no gradients.
         base_corr = G.base[0]
         displacement_corr = mic_displacement(coords.detach() - base_corr, box_size)
         unwrapped_corr = base_corr + displacement_corr
         centered0 = base_corr - base_corr.mean(0)
         centered1 = unwrapped_corr - unwrapped_corr.mean(0)
         numer = (centered0 * centered1).sum(0)
         denom = (centered0.square().sum(0) * centered1.square().sum(0)).sqrt()
         cx_value, cy_value = (numer / denom.clamp_min(1e-12)).cpu().tolist()
         global_abs_psi6 = torch.sqrt(psi_re.mean().square() + psi_im.mean().square()).item()
     writer.writerow([epoch + 1, loss.item(), L_hyper.item(),
                      L_repulsion.item(), L_smooth.item(), L_hex.item(),
                      L_hex_local.item(), L_hex_align.item(), psi_abs.mean().item(),
                      global_abs_psi6, cx_value, cy_value,
                      elapsed_before + time.perf_counter() - start_time])
     if (epoch + 1) % MEMORY_REPORT_EVERY == 0:
         report_memory(epoch + 1)
     if (epoch + 1) % CHECKPOINT_EVERY == 0:
         save_checkpoint(epoch + 1)

     if epoch % 50 == 0:
         print(f"Epoch {epoch}: Loss={loss.item():.4e}  "
               f"(hyper={L_hyper.item():.4e}, rep={L_repulsion.item():.4e}, "
               f"smooth={L_smooth.item():.4e}, hex={L_hex.item():.4e}, "
               f"weighted_hex={lambda_hex * L_hex.item():.4e}, "
               f"mean|psi6|={psi_abs.mean().item():.4f})")

     # -----------------------------
     # Snapshots every 200 epochs
     # -----------------------------
     if epoch % snapshot_every == 0:
         coords_sample = coords.detach().cpu().numpy()
         np.savetxt(f"{RUN_DIR}/snapshots/points_epoch_{epoch:05d}.csv", coords_sample, delimiter=",", header="x,y", comments="")
         if epoch % (2 if SMOKE else 500) == 0:
             p6 = np.column_stack((coords_sample, psi_re.detach().cpu().numpy(),
                                   psi_im.detach().cpu().numpy(), psi_abs.detach().cpu().numpy()))
             np.savetxt(f"{RUN_DIR}/snapshots/psi6_epoch_{epoch:05d}.csv", p6,
                        delimiter=",", header="x,y,psi6_real,psi6_imag,abs_psi6", comments="")
         plt.figure(figsize=(5,5))
         plt.scatter(coords_sample[:,0], coords_sample[:,1], s=1)
         plt.title(f"Points (epoch {epoch})")
         plt.xlim(0, box_size)
         plt.ylim(0, box_size)
         plt.gca().set_aspect('equal')
         plt.tight_layout()
         plt.savefig(f"{RUN_DIR}/snapshots/points_epoch_{epoch:05d}.png", dpi=160)
         plt.close()

         k_mag_vals, S_k_vals = structure_factor_np(coords_sample, n_k=n_k, box_size=box_size)
         k_centers, S_rad = radial_average(k_mag_vals, S_k_vals)
         plt.figure(figsize=(5,4))
         plt.plot(k_centers, S_rad, "o-", label="Generated")
         plt.xlabel("k")
         plt.ylabel("S(k)")
         plt.title(f"S(k) (epoch {epoch})")
         plt.grid(True)
         plt.legend()
         plt.tight_layout()
         plt.savefig(f"{RUN_DIR}/snapshots/Sk_epoch_{epoch:05d}.png", dpi=160)
         plt.close()

 # Final trained model evaluated at the LAST sampled latent z, but without
 # coordinate noise; z is recorded so the coordinates are exactly reproducible.
 # Unlike pre-update snapshots, this uses the weights AFTER the last Adam step.
 with torch.no_grad():
     final_coords = G(z.detach())[0].cpu().numpy()
     last_training_coords = coords.detach().cpu().numpy()
     final_latent = z.detach().cpu().numpy()
 np.savetxt(os.path.join(RUN_DIR, "final_generated_positions.csv"), final_coords,
            delimiter=",", header="x,y", comments="")
 np.savetxt(os.path.join(RUN_DIR, "last_training_sample_positions.csv"),
            last_training_coords, delimiter=",", header="x,y", comments="")
 np.savetxt(os.path.join(RUN_DIR, "final_latent_vector.csv"), final_latent,
            delimiter=",", comments="")
 save_checkpoint(epochs)
finally:
 logfile.close()
 sync_device()
 elapsed = elapsed_before + time.perf_counter() - start_time
 print(f"Total accumulated runtime: {elapsed:.1f} seconds ({elapsed / 3600:.2f} hours)", flush=True)
 with open(os.path.join(RUN_DIR, "runtime.txt"), "w") as f:
     f.write(f"Accumulated runtime (seconds): {elapsed:.2f}\\n")
