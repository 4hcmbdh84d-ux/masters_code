#!/usr/bin/env python3
"""Unattended, restart-safe hexatic lambda x beta x seed experiment queue.

Place THIS FILE beside the EXISTING (unchanged):
    ~/Downloads/overnight_scientific_hexatic.py
    ~/Downloads/dhu_hexatic_memory_safe.py

Dry run:   python3 -u run_hexatic_weekend_matrix.py --dry-run
Mac run:   caffeinate -i -s python3 -u run_hexatic_weekend_matrix.py
No email:  caffeinate -i -s python3 -u run_hexatic_weekend_matrix.py --no-email

The queue launches independent experiments sequentially with 15-minute idle
periods between training processes. It never trains its own model or edits the
two existing scripts. Gmail credentials stay in process memory.
"""
from __future__ import annotations

import argparse
import csv
from email.message import EmailMessage
import fcntl
import getpass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import smtplib
import subprocess
import sys
import time

BASE = Path(__file__).resolve().parent
TRAINER = BASE / "dhu_hexatic_memory_safe.py"
MANAGER = BASE / "overnight_scientific_hexatic.py"
ROOT = BASE / "hexatic_experiments"


def fmt(x: float) -> str:
    # Same folder formatting as the existing experiment manager.
    return f"{x:g}"


def folder_for(seed: int, lam: float, beta: float) -> Path:
    return ROOT / f"seed_{seed}_lambda_{fmt(lam)}_beta_{fmt(beta)}"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", type=int, nargs="+", default=[101, 73921, 4829])
    p.add_argument("--lambdas", type=float, nargs="+", default=[0.001, 0.005, 0.02])
    p.add_argument("--betas", type=float, nargs="+", default=[0.25, 1.0, 3.0])
    p.add_argument("--epochs", type=int, default=15000)
    p.add_argument("--cooldown-minutes", type=float, default=15.0,
                   help="GPU/CPU idle break before starting the next training process (default: 15; use 0 to disable)")
    p.add_argument("--retries", type=int, default=1,
                   help="Additional attempts if an individual run fails (default: 1)")
    p.add_argument("--notify-every", type=int, default=3,
                   help="Progress email after every N completed runs (default: 3)")
    p.add_argument("--min-free-gb", type=float, default=3.0,
                   help="Stop starting new jobs if free disk space falls below this")
    p.add_argument("--stop-on-error", action="store_true",
                   help="Stop queue after an unsuccessful run instead of continuing")
    p.add_argument("--no-email", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--check-only", action="store_true",
                   help="Run the manager's checks for every lambda/beta combination")
    a = p.parse_args(argv)
    if any(x < 0 or x > 4294967295 for x in a.seeds) or len(set(a.seeds)) != len(a.seeds):
        p.error("Seeds must be distinct integers from 0 to 2^32-1")
    for label, values in (("lambda", a.lambdas), ("beta", a.betas)):
        if any(not math.isfinite(x) or x < 0 for x in values):
            p.error(f"{label} values must be nonnegative and finite")
        if len(set(fmt(x) for x in values)) != len(values):
            p.error(f"{label} values must generate distinct experiment folders")
    if (a.epochs < 1 or a.retries < 0 or a.notify_every < 1 or
            not math.isfinite(a.min_free_gb) or a.min_free_gb < 0 or
            not math.isfinite(a.cooldown_minutes) or a.cooldown_minutes < 0):
        p.error("Invalid epochs, retries, notification frequency, disk threshold or cooldown")
    return a


def latest_epoch(folder: Path) -> int:
    history = folder / "loss_history.csv"
    if not history.is_file():
        return 0
    with history.open(newline="") as stream:
        return max((int(row["epoch"]) for row in csv.DictReader(stream)
                    if row.get("epoch")), default=0)


def verified_complete(folder: Path, seed: int, lam: float, beta: float,
                      epochs: int, expected_sha: str) -> bool:
    required = ["checkpoint_latest.pt", "final_generated_positions.csv",
                "final_metrics.json", "manifest.json", "loss_history.csv"]
    if any(not (folder / name).is_file() for name in required):
        return False
    manifest = json.loads((folder / "manifest.json").read_text())
    if manifest.get("trainer_sha256") != expected_sha:
        raise RuntimeError(f"Trainer SHA differs for existing completed run: {folder}")
    metrics = json.loads((folder / "final_metrics.json").read_text())
    for record in (manifest, metrics):
        if (record.get("seed") != seed or record.get("lambda_hex") != lam or
                record.get("beta") != beta):
            raise RuntimeError(f"Parameter mismatch in existing run: {folder}")
    return latest_epoch(folder) == epochs and metrics.get("epoch") == epochs


def save_status(path: Path, data: dict):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    os.replace(temporary, path)


def say(message: str, logfile: Path | None = None):
    print(message, flush=True)
    if logfile is not None:
        with logfile.open("a", encoding="utf-8") as out:
            out.write(message + "\n")


def command(seed: int, lam: float, beta: float, epochs: int) -> list[str]:
    return [sys.executable, "-u", str(MANAGER), "--seed", str(seed),
            "--lambda-hex", fmt(lam), "--beta", fmt(beta),
            "--epochs", str(epochs)]


def run_child(cmd: list[str], logfile: Path) -> int:
    """One child at a time; preserve output and Ctrl+C the whole child process group."""
    with logfile.open("a", encoding="utf-8", buffering=1) as out:
        out.write("\nCOMMAND: " + " ".join(cmd) + "\n")
        process = subprocess.Popen(cmd, cwd=BASE, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True,
                                   bufsize=1, start_new_session=True)
        try:
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                out.write(line)
            return process.wait()
        except KeyboardInterrupt:
            try:
                os.killpg(process.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait()
            raise


def wait_for_cooldown(last_training_end: float | None, minutes: float,
                      logfile: Path, status_path: Path, state: dict):
    """Keep the trainer fully stopped for a minimum idle interval between runs.

    Runs in the queue's main process, so caffeinate stays active. A delayed
    notification/verification already counts toward the interval. No pause
    happens before the first training process, after the last, or for skipped
    completed experiments.
    """
    if last_training_end is None or minutes <= 0:
        return
    deadline = last_training_end + 60.0 * minutes
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return
    say(f"[COOLDOWN] Trainer idle for {remaining/60:.1f} more minutes "
        f"(configured break: {minutes:g} min) before the next run", logfile)
    state["cooldown"] = {"status": "waiting", "minutes": minutes,
                         "remaining_seconds": math.ceil(remaining)}
    save_status(status_path, state)
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(60.0, remaining))
        remaining = max(0.0, deadline - time.monotonic())
        state["cooldown"]["remaining_seconds"] = math.ceil(remaining)
        save_status(status_path, state)
        if remaining > 0:
            say(f"[COOLDOWN] {remaining/60:.1f} min remaining", logfile)
    state["cooldown"].update(status="finished", remaining_seconds=0)
    save_status(status_path, state)
    say("[COOLDOWN] Finished; next experiment may start", logfile)


def send_email(auth, subject: str, body: str):
    sender, recipient, password = auth
    message = EmailMessage()
    message["From"], message["To"], message["Subject"] = sender, recipient, subject
    message.set_content(body)
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=30) as smtp:
        smtp.login(sender, password)
        smtp.send_message(message)


def main():
    a = parse_args()
    if not MANAGER.is_file() or not TRAINER.is_file():
        raise RuntimeError("Place this beside overnight_scientific_hexatic.py and dhu_hexatic_memory_safe.py")
    trainer_sha = hashlib.sha256(TRAINER.read_bytes()).hexdigest()
    manager_sha = hashlib.sha256(MANAGER.read_bytes()).hexdigest()
    # Use each seed with every pair: a paired comparison across parameters.
    jobs = [(seed, lam, beta) for seed in a.seeds
            for lam in a.lambdas for beta in a.betas]
    job_ids = [folder_for(*j).name for j in jobs]
    plan = dict(seeds=a.seeds, lambdas=a.lambdas, betas=a.betas, epochs=a.epochs)
    plan_hash = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()[:12]
    log = ROOT / f"weekend_matrix_{plan_hash}.log"
    status_path = ROOT / f"weekend_matrix_{plan_hash}_status.json"

    print(f"[PLAN] {len(jobs)} jobs; seeds={a.seeds}; lambda={a.lambdas}; beta={a.betas}")
    print(f"[PLAN] {a.epochs} epochs/job; idle break between runs={a.cooldown_minutes:g} min. "
          f"Current trainer SHA256={trainer_sha}")
    print(f"[LOG] {log}\n[STATUS] {status_path}")
    for number, (seed, lam, beta) in enumerate(jobs, start=1):
        print(f"  {number:02d}/{len(jobs)}: seed={seed}, lambda={fmt(lam)}, beta={fmt(beta)}")
    if a.dry_run:
        print("[DRY RUN] No experiments or files created.")
        return 0

    ROOT.mkdir(parents=True, exist_ok=True)
    # Prevent launching TWO weekend queues at once.
    with (ROOT / ".weekend_matrix.lock").open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another weekend matrix is already running")
        if a.check_only:
            for lam in a.lambdas:
                for beta in a.betas:
                    say(f"[CHECK] lambda={fmt(lam)} beta={fmt(beta)}", log)
                    if run_child(command(a.seeds[0], lam, beta, a.epochs) +
                                 ["--check-only"], log) != 0:
                        say("[FAILED] Preflight failed; no production runs launched", log)
                        return 1
            say("[PASS] All preflight tests passed; no production runs launched", log)
            return 0

        # Run actual CPU/MPS gradients and a 3-epoch smoke before soliciting Gmail.
        say("[PREFLIGHT] Checking trainer on this Mac...", log)
        if run_child(command(*jobs[0], a.epochs) + ["--check-only"], log) != 0:
            say("[HALT] Preflight failed. No full-length runs launched.", log)
            return 1
        auth = None
        if not a.no_email:
            sender = input("Gmail address: ").strip()
            recipient = input("Notification address [Enter for same]: ").strip() or sender
            password = getpass.getpass("Gmail App Password (hidden): ").replace(" ", "")
            if not sender or not password:
                raise RuntimeError("Gmail credentials required (or use --no-email)")
            auth = (sender, recipient, password)
            # Refuse to start unattended runs if notifications weren't tested.
            send_email(auth, "[Hexatic weekend] Test notification",
                       f"Notifications work. Planned jobs: {len(jobs)}\nStatus file: {status_path}")
            say("[EMAIL] Test message sent", log)

        def notify(subject: str, text: str):
            if auth is not None:
                try:
                    send_email(auth, subject, text)
                except Exception as exc:
                    say(f"[EMAIL WARNING] Notification failed: {exc}; jobs continue", log)

        state = {"plan": plan, "cooldown_minutes": a.cooldown_minutes,
                 "trainer_sha256": trainer_sha,
                 "manager_sha256": manager_sha, "order": job_ids,
                 "jobs": {key: {"status": "pending", "folder": str(folder_for(*job))}
                          for key, job in zip(job_ids, jobs)}}
        save_status(status_path, state)
        success, failures = 0, 0
        last_training_end = None
        start = time.monotonic()
        for index, (job, key) in enumerate(zip(jobs, job_ids), start=1):
            seed, lam, beta = job
            folder = folder_for(*job)
            try:
                done = verified_complete(folder, seed, lam, beta, a.epochs, trainer_sha)
            except (ValueError, OSError, KeyError) as exc:
                raise RuntimeError(f"Cannot verify existing run {folder}: {exc}") from exc
            if done:
                success += 1
                state["jobs"][key].update(status="completed_existing", epoch=a.epochs)
                save_status(status_path, state)
                say(f"[SKIP] {index}/{len(jobs)} already complete: {key}", log)
                continue

            free_gb = shutil.disk_usage(BASE).free / (1024 ** 3)
            if free_gb < a.min_free_gb:
                state["jobs"][key].update(status="blocked_low_disk", free_gb=round(free_gb, 2))
                save_status(status_path, state)
                msg = f"[HALT] Free disk {free_gb:.2f} GiB below {a.min_free_gb} GiB"
                say(msg, log)
                notify("[Hexatic weekend] Stopped: low disk space", msg)
                return 1

            finished = False
            for attempt in range(1, a.retries + 2):
                # Wait only between actual training subprocesses (including retries).
                # Already-completed jobs do not need a cooldown.
                wait_for_cooldown(last_training_end, a.cooldown_minutes,
                                  log, status_path, state)
                state["jobs"][key].update(status="running", attempt=attempt,
                                          epoch=latest_epoch(folder))
                save_status(status_path, state)
                say(f"[START] {index}/{len(jobs)} {key} attempt={attempt}/{a.retries+1}"
                    f" epoch={latest_epoch(folder)}/{a.epochs}", log)
                t0 = time.monotonic()
                try:
                    exit_code = run_child(command(*job, a.epochs), log)
                except KeyboardInterrupt:
                    state["jobs"][key].update(status="interrupted",
                                              epoch=latest_epoch(folder))
                    save_status(status_path, state)
                    notify("[Hexatic weekend] Interrupted", f"Interrupted: {key}")
                    raise
                # Time from when the child process exited, not from when the
                # next training job is scheduled or the status is written.
                last_training_end = time.monotonic()
                finished = exit_code == 0 and verified_complete(
                    folder, seed, lam, beta, a.epochs, trainer_sha)
                state["jobs"][key].update(exit_code=exit_code,
                    epoch=latest_epoch(folder), last_attempt_minutes=round(
                        (time.monotonic() - t0) / 60, 1))
                save_status(status_path, state)
                if finished:
                    break
                say(f"[RETRY] {key}: unsuccessful attempt {attempt}", log)

            if finished:
                success += 1
                metrics = json.loads((folder / "final_metrics.json").read_text())
                state["jobs"][key].update(status="completed", epoch=a.epochs,
                    mean_abs_psi6=metrics.get("mean_abs_psi6"),
                    global_abs_psi6=metrics.get("global_abs_psi6"),
                    hyper=metrics.get("hyper"),
                    hex_local=metrics.get("hex_local"),
                    hex_align=metrics.get("hex_align"))
                say(f"[DONE] {index}/{len(jobs)} {key}: "
                    f"mean|psi6|={metrics.get('mean_abs_psi6', float('nan')):.4f}, "
                    f"global|psi6|={metrics.get('global_abs_psi6', float('nan')):.4f}", log)
                if success % a.notify_every == 0:
                    notify(f"[Hexatic weekend] {success}/{len(jobs)} completed",
                           f"Most recent: {key}\nStatus: {status_path}")
            else:
                failures += 1
                state["jobs"][key].update(status="failed", epoch=latest_epoch(folder))
                msg = f"[FAILED] {index}/{len(jobs)}: {key}; continuing={not a.stop_on_error}"
                say(msg, log)
                notify("[Hexatic weekend] Experiment failed", msg)
                if a.stop_on_error:
                    save_status(status_path, state)
                    return 1
            save_status(status_path, state)

        elapsed_h = (time.monotonic() - start) / 3600
        summary = f"[FINISHED] {success}/{len(jobs)} complete, {failures} failed, {elapsed_h:.2f} h elapsed"
        say(summary, log)
        notify("[Hexatic weekend] Queue finished", summary + f"\nStatus: {status_path}")
        return 0 if failures == 0 else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("[STOP] Interrupted; rerun the same command to resume from checkpoints", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"[HALTED] {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
