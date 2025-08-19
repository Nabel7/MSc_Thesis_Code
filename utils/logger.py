
# utils/logger.py
# Minimal CSV logger for episode-level (and optional step-level) metrics.
from __future__ import annotations
import csv
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, Any

@dataclass
class EpisodeLog:
    split: str                 # "train" or "val"
    episode: int
    steps: int
    ep_return: float
    clear_rate_buy: float
    clear_rate_sell: float
    loss: float
    eps: float
    beta: float

class CSVLogger:
    def __init__(self, episode_csv: str, step_csv: Optional[str] = None):
        self.episode_csv = Path(episode_csv)
        self.step_csv = Path(step_csv) if step_csv else None
        # Create files with header if new
        if not self.episode_csv.exists():
            self.episode_csv.parent.mkdir(parents=True, exist_ok=True)
            with self.episode_csv.open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["split","episode","steps","ep_return","clear_rate_buy","clear_rate_sell","loss","eps","beta"])
        if self.step_csv and (not self.step_csv.exists()):
            self.step_csv.parent.mkdir(parents=True, exist_ok=True)
            with self.step_csv.open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    "split","episode","t","DAM","BM","ref","side","q","delta","limit","cleared","reward"
                ])

    def log_episode(self, e: EpisodeLog) -> None:
        with self.episode_csv.open("a", newline="") as f:
            w = csv.writer(f)
            w.writerow([e.split, e.episode, e.steps, e.ep_return, e.clear_rate_buy, e.clear_rate_sell, e.loss, e.eps, e.beta])

    def log_step(self,
                 split: str, episode: int, t: int,
                 DAM: float, BM: float, ref: float,
                 side: int, q: float, delta: float, limit: float,
                 cleared: bool, reward: float) -> None:
        """Optional: call each env.step to build behavior diagnostics (can be large)."""
        if self.step_csv is None:
            return
        with self.step_csv.open("a", newline="") as f:
            w = csv.writer(f)
            w.writerow([split, episode, t, DAM, BM, ref, side, q, delta, limit, int(bool(cleared)), reward])
