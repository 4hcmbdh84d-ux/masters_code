#!/usr/bin/env python3
"""Validated one-seed run of the new hexatic trainer.

Based on user's overnight_scientific_replicates.py:
- Same dense-vs-blockwise repulsion gradient validation.
- Added independent psi6 geometry / gradient validation.
- Isolated 3-epoch smoke test before any production run.
- One fresh independent seed (default 101) rather than running 101/202/303.
- Separate output directories; original fausto_replicates/ and overnight_run/ untouched.
- Checkpoint resume; source hash & parameter manifest prevent silent changes.
"""
from __future__ import annotations
import argparse
import ast
import csv
import fcntl
import hashlib
import json
import math
import os
import platform
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import scipy
from scipy.spatial import cKDTree
import torch

BASE = Path(__file__).resolve().parent
SOURCE = BASE / "dhu_hexatic_memory_safe.py"
ROOT = BASE / "hexatic_experiments"


def reference_repulsion(coords, box_length, cutoff):
    """Unchanged dense reference from user's replicate launcher."""
    delta = coords.unsqueeze(0) - coords.unsqueeze(1)
    delta = delta - torch.round(delta / box_length) * box_length
    d2 = (delta ** 2).sum(dim=-1) + 1e-6
    mask = d2 < cutoff**2
    mask.fill_diagonal_(False)
    inv = torch.where(mask, 1.0 / d2, torch.zeros_like(d2))
    n = mask.sum()
    return inv.sum() / n.clamp(min=1)


def load_actual_functions(source_text):
    tree = ast.parse(source_text)
    needed = {"BlockwiseRepulsion", "mic_displacement", "periodic_six_neighbours", "hexatic_order_loss"}
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in needed]
    found = {n.name for n in nodes}
    if found != needed:
        raise RuntimeError(f"Trainer missing test functions: {needed - found}")
    namespace = {"torch": torch, "np": np, "cKDTree": cKDTree}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace


def check_repulsion_gradients(blockwise):
    """The exact test from overnight_scientific_replicates, for CPU and MPS."""
    print("[CHECK] Comparing original dense repulsion vs memory-safe implementation", flush=True)
    report = []
    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.append("mps")
    for dev in devices:
        for n in (48, 128, 256):
            for trialseed in (13, 29):
                L = 10.0 * math.sqrt(n)
                g = torch.Generator().manual_seed(trialseed + n)
                positions = torch.rand(n, 2, generator=g) * L
                positions[0] = torch.tensor([L - 0.3, 0.4])
                positions[1] = torch.tensor([0.5, L - 0.6])
                a = positions.to(dev).requires_grad_(True)
                b = positions.to(dev).clone().detach().requires_grad_(True)
                dense_loss = reference_repulsion(a, L, 25.0)
                block_loss = blockwise.apply(b, L, 25.0, 32)
                dense_loss.backward()
                block_loss.backward()
                if dev == "mps":
                    torch.mps.synchronize()
                lv, bv = float(dense_loss.detach().cpu()), float(block_loss.detach().cpu())
                ga, gb = a.grad.detach().cpu(), b.grad.detach().cpu()
                max_grad_error = float((ga - gb).abs().max())
                rel_grad = float(torch.linalg.vector_norm(ga-gb) / torch.linalg.vector_norm(ga).clamp(min=1e-12))
                rel_loss = abs(lv-bv) / max(abs(lv), 1e-12)
                passed = rel_loss < 2e-4 and rel_grad < 2e-3 and max_grad_error < 2e-5
                report.append({"device": dev, "N": n, "seed": trialseed,
                               "loss_dense": lv, "loss_blockwise": bv,
                               "rel_loss_diff": rel_loss, "rel_grad_l2_diff": rel_grad,
                               "max_abs_grad_diff": max_grad_error, "pass": passed})
                print(f"  {dev:3s} N={n:3d} case={trialseed} | "
                      f"loss rel error={rel_loss:.2e}; grad rel error={rel_grad:.2e}: "
                      f"{'PASS' if passed else 'FAIL'}", flush=True)
                if not passed:
                    raise RuntimeError("Dense vs blockwise repulsion gradient check failed")
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "gradient_validation.json").write_text(json.dumps(report, indent=2))
    print(f"[CHECK] {len(report)} repulsion tests passed", flush=True)


def check_hexatic(functions):
    psi_loss = functions["hexatic_order_loss"]
    six_neighbours = functions["periodic_six_neighbours"]
    print("[CHECK] Testing sixfold geometry, periodic six-neighbour selection and gradients", flush=True)
    # Central particle, exactly six regularly spaced neighbours.
    angles = np.arange(6) * np.pi/3
    regular = np.vstack([np.array([[50.0,50.0]]),
                         np.column_stack((50+8*np.cos(angles), 50+8*np.sin(angles)))])
    coords = torch.tensor(regular, dtype=torch.float64, requires_grad=True)
    total, loc, ali, r, im, mags = psi_loss(coords, 100., 0.25)
    assert float(mags[0].detach()) > 0.999999, f"Regular hexagon center gave psi6={mags[0]}"
    assert torch.isfinite(total), "Nonfinite hexatic loss on regular hexagon"
    # Cross-boundary neighbour (closest separation is 0.3, not 99.7).
    rng = np.random.default_rng(345)
    x = rng.uniform(5,95,(24,2))
    x[0] = [99.8,50]
    x[1] = [0.1,50]
    indices = six_neighbours(torch.tensor(x),100.).cpu().numpy()
    assert 1 in indices[0], "Periodic boundary nearest-neighbour search failed"
    assert not np.any(indices==np.arange(len(x))[:,None]), "Self-neighbour incorrectly included"
    assert np.all(np.diff(np.sort(indices,axis=1),axis=1)>0), "Duplicate neighbour found"
    # Torch autograd gradient checking (neighbour topology stable to epsilon).
    z = rng.uniform(5,95,(20,2))
    t = torch.tensor(z,dtype=torch.float64,requires_grad=True)
    def fn(a):
        return psi_loss(a,100.,0.25)[0]
    passed = torch.autograd.gradcheck(fn,(t,),eps=1e-6,atol=1e-5,rtol=1e-4)
    assert passed, "Gradient check failed for hexatic loss"
    print("  CPU float64 hexatic-loss gradient check PASS", flush=True)
    devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
    for device in devices:
        t = torch.tensor(z,dtype=torch.float32,device=device,requires_grad=True)
        out = psi_loss(t,100.,0.25)[0]
        grad = torch.autograd.grad(out,t)[0]
        assert bool(torch.isfinite(grad).all()), f"Nonfinite psi6 gradient on {device}"
        assert float(grad.abs().sum()) > 0, f"Zero psi6 gradient on {device}"
        print(f"  {device} finite nonzero hexatic gradient PASS", flush=True)
    print("[CHECK] Hexatic geometry and gradient tests passed", flush=True)


def experiment_dir(seed,lam,beta):
    return ROOT / f"seed_{seed}_lambda_{lam:g}_beta_{beta:g}"


def env_for(seed, lam, beta, epochs, smoke=False):
    env = os.environ.copy()
    env.update(HEX_SEED=str(999 if smoke else seed), HEX_LAMBDA=str(lam),
               HEX_BETA=str(beta), HEX_SMOKE="1" if smoke else "0", HEX_EPOCHS=str(epochs),
               HEX_SMOKE_TAG=hashlib.sha256(SOURCE.read_bytes()).hexdigest()[:12] if smoke else "")
    return env


def latest_epoch(folder):
    history = folder / "loss_history.csv"
    if not history.exists():
        return 0
    with history.open(newline="") as f:
        return max((int(r["epoch"]) for r in csv.DictReader(f) if r.get("epoch")), default=0)


def run_smoke(args):
    folder = ROOT / f"_smoke_{hashlib.sha256(SOURCE.read_bytes()).hexdigest()[:12]}_seed_999_lambda_{args.lambda_hex:g}_beta_{args.beta:g}"
    if latest_epoch(folder) < 3:
        log_path = ROOT / "smoke_training.log"
        print("[CHECK] Starting isolated three-epoch small-system smoke test", flush=True)
        with log_path.open("w",encoding="utf-8") as log:
            p = subprocess.run([sys.executable,"-u",str(SOURCE)],env=env_for(999,args.lambda_hex,args.beta,3,smoke=True),
                               cwd=BASE,stdout=log,stderr=subprocess.STDOUT)
        if p.returncode != 0:
            raise RuntimeError("Smoke test failed; see " + str(log_path) + ":\n" + log_path.read_text()[-2500:])
    with (folder / "loss_history.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    needed = {"hex","hex_local","hex_align","mean_abs_psi6","Cx_unwrapped","Cy_unwrapped"}
    if len(rows)!=3 or not needed.issubset(rows[0]) or not (folder/"final_generated_positions.csv").exists():
        raise RuntimeError("Smoke test outputs missing or malformed")
    print("[CHECK] Three-epoch full pipeline smoke test PASSED", flush=True)


def _run_experiment_locked(args, source_text):
    folder = experiment_dir(args.seed,args.lambda_hex,args.beta)
    folder.mkdir(parents=True,exist_ok=True)
    source_hash = hashlib.sha256(source_text.encode()).hexdigest()
    manifest = folder/"manifest.json"
    config = {"seed":args.seed,"lambda_hex":args.lambda_hex,"beta":args.beta,
              "trainer_sha256":source_hash,"original_three_loss_weights":[1.0,1.0,1.0],
              "n_particles":10000,"box_size":1000.0,
              "python":sys.version,"torch":torch.__version__,"scipy":scipy.__version__,
              "system":platform.platform(),
              "notes":"Four-loss experimental variant; original baselines untouched"}
    if manifest.exists():
        saved = json.loads(manifest.read_text())
        for key in ("seed","lambda_hex","beta","trainer_sha256"):
            if saved[key]!=config[key]:
                raise RuntimeError(f"Refusing to resume: manifest mismatch in {key} for {folder}")
    else:
        if (folder/"checkpoint_latest.pt").exists() or (folder/"loss_history.csv").exists():
            raise RuntimeError("Run files exist without manifest; refusing unsafe resume")
        manifest.write_text(json.dumps(config,indent=2))
    start = latest_epoch(folder)
    if start > args.epochs:
        raise RuntimeError(f"Existing run already at epoch {start}, requested only {args.epochs}")
    print(f"[RUN] Seed={args.seed}, lambda_hex={args.lambda_hex}, beta={args.beta}", flush=True)
    print(f"[RUN] Epochs {start} -> {args.epochs}, directory {folder}",flush=True)
    if start < args.epochs:
        with (folder/"training.log").open("a",buffering=1,encoding="utf-8") as log:
            log.write(f"\n=== START/RESUME: from logged epoch {start}, target {args.epochs} ===\n")
            t = time.monotonic()
            p = subprocess.run([sys.executable,"-u",str(SOURCE)],
                    env=env_for(args.seed,args.lambda_hex,args.beta,args.epochs),
                    cwd=BASE,stdout=log,stderr=subprocess.STDOUT)
        done = latest_epoch(folder)
        print(f"[RUN] Exit={p.returncode}, logged epoch={done}/{args.epochs}, "
              f"this session took {(time.monotonic()-t)/60:.1f} min",flush=True)
        if p.returncode != 0 or done != args.epochs or not (folder/"checkpoint_latest.pt").is_file():
            print("[ERROR] Last 25 log lines:\n"+"\n".join((folder/"training.log").read_text(errors="replace").splitlines()[-25:]),flush=True)
            raise RuntimeError("Run stopped before completion. Re-run same command to resume from checkpoint.")
    else:
        if not (folder/"checkpoint_latest.pt").is_file():
            raise RuntimeError("Log reports completion but no checkpoint is present")
        print("[RUN] Already completed this target; skipping retraining",flush=True)
    with (folder/"loss_history.csv").open(newline="") as f:
        last = list(csv.DictReader(f))[-1]
    summary = {"seed":args.seed,"epoch":int(last["epoch"]),
               "lambda_hex":args.lambda_hex,"beta":args.beta,
               **{k:float(last[k]) for k in last if k!="epoch"}}
    (folder/"final_metrics.json").write_text(json.dumps(summary,indent=2))
    print(f"[SUCCESS] Final loss={summary['total']:.6g}; "
          f"mean|psi6|={summary['mean_abs_psi6']:.4f}; final output: {folder}",flush=True)
    print("[INFO] Analyse final coordinates with: python3 analyse_hexatic_run.py",flush=True)


def run_experiment(args, source_text):
    # Advisory POSIX lock: prevent two launchers from overwriting same run.
    folder = experiment_dir(args.seed, args.lambda_hex, args.beta)
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / ".run.lock").open("a+") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another process is already managing this run: {folder}") from exc
        try:
            _run_experiment_locked(args, source_text)
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed",type=int,default=101,help="Independent seed (default 101)")
    ap.add_argument("--lambda-hex",type=float,default=0.005,help="New fourth-loss coefficient")
    ap.add_argument("--beta",type=float,default=0.25,help="Hexatic neighbour-alignment coefficient")
    ap.add_argument("--epochs",type=int,default=15000,help="Total target epochs, can extend on resume")
    ap.add_argument("--check-only",action="store_true",help="Validate including smoke test, do not start full run")
    args=ap.parse_args()
    if args.seed<0 or args.epochs<1 or args.lambda_hex<0 or args.beta<0:
        ap.error("Seed and weights must be nonnegative; epochs must be positive")
    if not SOURCE.exists():
        raise FileNotFoundError(f"Place dhu_hexatic_memory_safe.py alongside this file: {SOURCE}")
    source_text=SOURCE.read_text()
    functions=load_actual_functions(source_text)
    check_repulsion_gradients(functions["BlockwiseRepulsion"])
    check_hexatic(functions)
    run_smoke(args)
    if args.check_only:
        print("[CHECK-ONLY] All validation passed; no 10,000-particle run launched.",flush=True)
        return
    run_experiment(args,source_text)


if __name__=="__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("[INTERRUPTED] Run the same command to resume from checkpoint",file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(f"[HALTED SAFELY] {exc}",file=sys.stderr,flush=True)
        sys.exit(1)
