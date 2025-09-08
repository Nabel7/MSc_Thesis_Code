# envs/generator_env.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import numpy as np
import pandas as pd
import math

class _Box:
    def __init__(self, low, high, shape):
        self.low = np.array([low], dtype=np.float32) if np.isscalar(low) else np.asarray(low, np.float32)
        self.high = np.array([high], dtype=np.float32) if np.isscalar(high) else np.asarray(high, np.float32)
        self.shape = shape

# ---- Gas unit parameters (hours + MW/h so dt is clear) ----
@dataclass
class PlantParams:
    P_min: float = 50.0
    P_max: float = 400.0
    ramp_up: float = 60.0
    ramp_down: float = 60.0
    min_up_steps: int = 8            # was 4
    min_down_steps: int = 8          # was 4
    start_cost: float = 10_000.0
    no_load_cost_per_hour: float = 600.0
    ramp_cost_per_MW: float = 2.0    # was 0.0 -> penalize dithering

@dataclass
class CostParams:
    variable_cost_per_MWh: float = 45.0   # € / MWh (can override from CSV outside if you want)

@dataclass
class EnvConfig:
    dt_hours: float = 0.25                 # 15 minutes
    episode_len: int = 96                  # 1 day = 96 x 15min
    normalize_features: bool = True
    add_time_features: bool = True
    time_col: Optional[str] = "DeliveryPeriod"  # try to use this if present
    reward_scale: float = 1e-3          # <-- add
    reward_clip: float = 5_000.0        # <-- add (clip € before scaling)

class ISEMGeneratorEnv:
    """
    Price-taker generator dispatch env with realistic unit constraints (I-SEM-ish).
    Action = set-point P_cmd in MW; env enforces ramp & min up/down and computes profit.
    Revenue currently = DAM * P * dt. (Easy to extend to imbalance later.)
    """
    def __init__(
        self,
        csv_path: str | Path,
        price_col: str = "DAM",
        feature_cols: Optional[List[str]] = None,
        plant: PlantParams = PlantParams(),
        costs: CostParams = CostParams(),
        cfg: EnvConfig = EnvConfig(),
    ):
        self.csv_path = str(csv_path)
        self.price_col = price_col
        self.feature_cols = list(feature_cols) if feature_cols else []
        self.plant = plant
        self.costs = costs
        self.cfg = cfg

        # ---- Load & time parsing ----
        self.df = pd.read_csv(self.csv_path)
        self.time = None
        tcol = cfg.time_col if (cfg.time_col and cfg.time_col in self.df.columns) else None
        if tcol is None:
            # heuristic: pick first datetime-like col if exists
            for c in self.df.columns:
                try:
                    parsed = pd.to_datetime(self.df[c])
                    if parsed.notna().any():
                        tcol = c
                        break
                except Exception:
                    pass
        if tcol:
            self.df[tcol] = pd.to_datetime(self.df[tcol], errors="coerce")
            self.time = self.df[tcol]

        if price_col not in self.df:
            raise ValueError(f"price_col '{price_col}' not found in {self.csv_path}")
        self.price = self.df[price_col].astype(float).values

        # ---- Build features ----
        if not self.feature_cols:
            # default: all numeric except price & time
            exclude = {price_col}
            if tcol: exclude.add(tcol)
            self.feature_cols = [c for c in self.df.columns
                                 if c not in exclude and np.issubdtype(self.df[c].dtype, np.number)]
        X = self.df[self.feature_cols].astype(float).values

        # add time features
        if self.cfg.add_time_features and self.time is not None:
            hour = self.time.dt.hour.to_numpy()
            dow = self.time.dt.dayofweek.to_numpy()
            X = np.hstack([
                X,
                np.cos(2*np.pi*hour/24.0)[:, None],
                np.sin(2*np.pi*hour/24.0)[:, None],
                np.cos(2*np.pi*dow/7.0)[:, None],
                np.sin(2*np.pi*dow/7.0)[:, None],
            ])
            self._time_feats = 4
        else:
            self._time_feats = 0

        # scale
        self.mean = X.mean(axis=0) if self.cfg.normalize_features else np.zeros(X.shape[1])
        self.std = X.std(axis=0) + 1e-8 if self.cfg.normalize_features else np.ones(X.shape[1])
        self.X_raw = X
        self.X = (X - self.mean) / self.std

        # runtime
        self.n = len(self.df)
        self.t = 0
        self.step_in_ep = 0

        # spaces
        self.action_space = _Box(0.0, float(self.plant.P_max), shape=(1,))
        self.obs_dim = self.X.shape[1] + 2  # features + [prev_P/Pmax, on_flag]

        # unit state
        self.P_prev = 0.0
        self.on = False

        # convert hours to steps given dt
        self._min_up_steps = max(1, int(round(self.plant.min_up_steps / self.cfg.dt_hours)))
        self._min_down_steps = max(1, int(round(self.plant.min_down_steps / self.cfg.dt_hours)))
        self.min_up_counter = 0
        self.min_down_counter = 0

        # per-step ramp limits (MW per step)
        self._ramp_up = self.plant.ramp_up * self.cfg.dt_hours
        self._ramp_down = self.plant.ramp_down * self.cfg.dt_hours

    # ---------------- core API ----------------
    def reset(self, *, start_index: Optional[int] = None) -> np.ndarray:
        if start_index is None:
            hi = self.n - self.cfg.episode_len - 1
            start_index = int(np.random.randint(0, max(1, hi)))
        self.t0 = start_index
        self.t = start_index
        self.step_in_ep = 0

        # random feasible start
        self.on = bool(np.random.rand() < 0.5)
        self.P_prev = self.plant.P_min if self.on else 0.0
        self.min_up_counter = np.random.randint(0, self._min_up_steps) if self.on else 0
        self.min_down_counter = 0 if self.on else np.random.randint(0, self._min_down_steps)

        return self._obs()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        # desired setpoint
        P_cmd = float(np.clip(action[0], self.action_space.low[0], self.action_space.high[0]))

        # ramped setpoint
        P_ramp_min = max(0.0, self.P_prev - self._ramp_down)
        P_ramp_max = min(self.plant.P_max, self.P_prev + self._ramp_up)
        P_star = float(np.clip(P_cmd, P_ramp_min, P_ramp_max))

        started = False
        shut = False

        # commitment logic with min up/down
        if self.on:
            if self.min_up_counter <= 0 and P_star < 0.5 * self.plant.P_min:
                self.on = False
                shut = True
                self.min_down_counter = self._min_down_steps
                P = 0.0
            else:
                if self.min_up_counter > 0 and P_star < self.plant.P_min:
                    P = self.plant.P_min
                else:
                    P = max(P_star, self.plant.P_min) if self.min_up_counter > 0 else P_star
                self.min_up_counter = max(0, self.min_up_counter - 1)
        else:
            if self.min_down_counter <= 0 and P_star >= 0.5 * self.plant.P_min:
                self.on = True
                started = True
                self.min_up_counter = self._min_up_steps
                P = max(P_star, self.plant.P_min)
            else:
                P = 0.0
                self.min_down_counter = max(0, self.min_down_counter - 1)

        price = float(self.price[self.t])
        dt = self.cfg.dt_hours

        revenue = price * P * dt
        var_cost = self.costs.variable_cost_per_MWh * P * dt
        no_load = (self.plant.no_load_cost_per_hour * dt) if self.on and P > 0 else 0.0
        startup = self.plant.start_cost if started else 0.0
        ramp_cost = self.plant.ramp_cost_per_MW * abs(P - self.P_prev)
        profit = revenue - var_cost - no_load - startup - ramp_cost

        info = dict(
            price=price, P=P, P_cmd=P_cmd, on=int(self.on), started=int(started), shut=int(shut),
            revenue=revenue, var_cost=var_cost, no_load=no_load,
            startup=startup, ramp_cost=ramp_cost, profit=profit
        )

        profit_clamped = float(np.clip(profit, -self.cfg.reward_clip, self.cfg.reward_clip))
        reward = profit_clamped * self.cfg.reward_scale

        self.P_prev = P
        self.t += 1
        self.step_in_ep += 1
        done = self.step_in_ep >= self.cfg.episode_len
        return self._obs(), float(reward), bool(done), info

    # ---------------- helpers ----------------
    def _obs(self) -> np.ndarray:
        x = self.X[self.t].astype(np.float32)
        aug = np.array([self.P_prev / max(1e-6, self.plant.P_max), 1.0 if self.on else 0.0], dtype=np.float32)
        return np.concatenate([x, aug], axis=0)

    @property
    def action_low(self) -> float: return float(self.action_space.low[0])
    @property
    def action_high(self) -> float: return float(self.action_space.high[0])
