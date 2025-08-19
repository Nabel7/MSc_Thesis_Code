# envs/speculator_env.py
# -----------------------------------------------------------------------------
# SpeculatorEnv: data-driven RL environment for DAM↔BM arbitrage.
#
# THEORY → CODE (MDP mapping)
#   • State s_t  : row features from the CSV at time t.
#   • Action a_t : limit price we submit.
#                  - By default: absolute €/MWh in [bid_low, bid_high].
#                  - Optional: delta around DAM, δ ∈ [-Δ, +Δ], with limit = DAM + δ.
#   • Transition : next row in the time-sorted CSV (exogenous dynamics).
#   • Reward r_t : realized P&L if the order clears in DAM, minus fees, scaled down.
#                  r_t = scale * 1[clear] * ( (2*side-1) * (BM - DAM) * q - fees )
#                  side=1 → buy (DAM→BM); side=0 → sell (DAM←BM).
# -----------------------------------------------------------------------------



import numpy as np
import pandas as pd
from types import SimpleNamespace
from types import SimpleNamespace
from typing import Optional, Iterable, Sequence, Dict, Any

class Space(SimpleNamespace):
    """Lightweight stand-in for gym.spaces.Box so the agent can read bounds/shapes."""
    def __init__(self, low, high, shape):
        super().__init__(low=np.array(low, dtype=np.float32),
                         high=np.array(high, dtype=np.float32),
                         shape=tuple(shape))

class SpeculatorEnv:
    """
    Environment encoding DAM↔BM arbitrage with a simple clearing rule.

    Parameters
    ----------
    csv_path : str
        Path to your dataset (must include 'DeliveryPeriod').
    feature_cols : list[str]
        Columns forming the observation vector s_t (e.g., forecasts, lags, time features).
        TIP: include 'side' and 'abs_volume' as features so the policy can condition on them.
    price_col : str
        Day-Ahead price column (EUR/MWh). Default: 'EURPrices'.
    bm_price_col : str
        Balancing Market price column (EUR/MWh). Default: 'BMImbalancePrice'.
    side_col : str
        Side indicator (0 = sell, 1 = buy). Determines clearing rule and P&L sign.
    volume_col : str
        Executed quantity in MWh. If missing, falls back to |ExecQty_MWh|, else unit volume.
    exclude_years : Iterable[int]
        Years to drop (e.g., (2023,) if no speculator data there).
    bid_low, bid_high : float
        Action bounds when using absolute price actions.
    episode_len : int
        Steps per episode (sampled contiguous block).
    seed : int
        RNG seed for start index sampling.
    per_mwh_fee : float
        Variable fee in €/MWh (applied only when you clear).
    fixed_fee : float
        Fixed fee in € per trade (applied only when you clear).
    reward_scale : float
        Scales raw P&L to keep gradients stable (e.g., 0.01).
    normalize_features : bool
        If True, z-score the feature columns using global mean/std (simple & fast).
        (For publication-grade experiments, compute stats on training folds only.)
    add_time_features : bool
        If True, auto-add sin/cos of hour and day-of-week to capture seasonality.
    use_delta_actions : bool
        If True, the action is δ in [-delta_max, +delta_max] and limit = DAM + δ.
        This often learns faster: buys → δ>0, sells → δ<0.
    delta_max : float
        Max absolute delta when use_delta_actions=True (in €/MWh).

    Observation
    -----------
    np.ndarray(shape=(len(feature_cols),), dtype=float32)

    Action
    ------
    • Absolute mode (default): np.ndarray([limit_price]) in [bid_low, bid_high]
    • Delta mode             : np.ndarray([delta])       in [-delta_max, +delta_max]

    Reward
    ------
    reward = reward_scale * 1[clear] * ( (2*side-1)*(BM - DAM)*q - (per_mwh_fee*q + fixed_fee) )
    """

    def __init__(self,
                 csv_path: Optional[str] = None,
                 df: Optional[pd.DataFrame] = None,
                 start: Optional[str] = None,
                 end: Optional[str] = None,
                 *,
                 feature_cols,
                 price_col: str = "EURPrices",
                 bm_price_col: str = "BMImbalancePrice",
                 side_col: str = "side",
                 volume_col: Optional[str] = None,
                 exclude_years: Iterable[int] = (2023,),
                 bid_low: float = 0.0,
                 bid_high: float = 300.0,
                 episode_len: int = 96,
                 seed: int = 0,
                 per_mwh_fee: float = 0.0,
                 fixed_fee: float = 0.0,
                 add_time_features: bool = True,
                 reward_scale: float = 0.01,
                 normalize_features: bool = True,
                 norm_stats: Optional[Dict[str, Any]] = None,   # <- NEW (μ/σ dict)
                 fit_normalizer: bool = False,                  # <- NEW (fit on this data)
                 use_delta_actions: bool = True,
                 delta_max: float = 50.0,
                 ref_price_col: str = "-24"):

        # -------- Load or use provided df --------
        if df is None:
            if csv_path is None:
                raise ValueError("Provide either csv_path or a prebuilt DataFrame via df=.")
            df = pd.read_csv(csv_path)

        if "DeliveryPeriod" not in df.columns:
            raise ValueError("CSV must include a 'DeliveryPeriod' column.")

        df = df.copy()
        df["DeliveryPeriod"] = pd.to_datetime(df["DeliveryPeriod"], errors="coerce")
        df = df.sort_values("DeliveryPeriod").reset_index(drop=True)

        # Optional global year exclusion
        if exclude_years:
            years = df["DeliveryPeriod"].dt.year
            df = df[~years.isin(list(exclude_years))]

        # Optional date window
        if (start is not None) or (end is not None):
            m = pd.Series(True, index=df.index)
            if start: m &= (df["DeliveryPeriod"] >= pd.to_datetime(start))
            if end:   m &= (df["DeliveryPeriod"] <= pd.to_datetime(end))
            df = df.loc[m]

        df = df.reset_index(drop=True)

        # Build feature list & add time features
        feats = list(feature_cols)
        if add_time_features:
            hr  = df["DeliveryPeriod"].dt.hour.values.astype(np.float32)
            dow = df["DeliveryPeriod"].dt.dayofweek.values.astype(np.float32)
            df["sin_hour"] = np.sin(2*np.pi*hr/24.0).astype(np.float32)
            df["cos_hour"] = np.cos(2*np.pi*hr/24.0).astype(np.float32)
            df["sin_dow"]  = np.sin(2*np.pi*dow/7.0).astype(np.float32)
            df["cos_dow"]  = np.cos(2*np.pi*dow/7.0).astype(np.float32)
            for extra in ["sin_hour", "cos_hour", "sin_dow", "cos_dow"]:
                if extra not in feats:
                    feats.append(extra)

        # Leakage guard
        for leak in [price_col, bm_price_col]:
            if leak in feats:
                raise ValueError(
                    f"Leakage risk: '{leak}' is in feature_cols. "
                    "Keep only pre-gate features (lags/forecasts/calendar/side)."
                )

        # Delta mode requires a known pre-gate reference
        self.use_delta_actions = bool(use_delta_actions)
        self.delta_max = float(delta_max)
        if self.use_delta_actions:
            if ref_price_col not in df.columns:
                raise ValueError(f"ref_price_col '{ref_price_col}' not found in data.")
            REF = df[ref_price_col].astype(np.float32).values
        else:
            REF = None

        # Volume selection / fallback
        if volume_col and (volume_col in df.columns):
            use_vol_col = volume_col
        elif "ExecQty_MWh" in df.columns:
            df["abs_volume_autogen"] = df["ExecQty_MWh"].abs()
            use_vol_col = "abs_volume_autogen"
        else:
            df["unit_volume"] = 1.0
            use_vol_col = "unit_volume"

        # Required columns
        required = set(feats + [price_col, bm_price_col, side_col, use_vol_col])
        if self.use_delta_actions:
            required.add(ref_price_col)
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in CSV: {missing}")

        # Drop NAs used in pipeline
        df = df.dropna(subset=list(required)).reset_index(drop=True)

        # Cache reward mechanics arrays
        DAM  = df[price_col].astype(np.float32).values
        BM   = df[bm_price_col].astype(np.float32).values
        SIDE = df[side_col].astype(np.int32).values
        Q    = df[use_vol_col].astype(np.float32).values

        # Build feature matrix (proper normalization)
        X_raw = df[feats].astype(np.float32)

        if normalize_features:
            if norm_stats is not None:
                # use provided μ/σ (train stats)
                mu = pd.Series(norm_stats["mu"]).astype(np.float32)
                sd = pd.Series(norm_stats["sd"]).astype(np.float32).replace(0, 1)
                # Align in case order differs
                mu = mu.reindex(X_raw.columns)
                sd = sd.reindex(X_raw.columns).replace(0, 1)
                X = ((X_raw - mu) / sd).astype(np.float32).values
                norm_stats_used = {"mu": mu.to_dict(), "sd": sd.to_dict()}
            elif fit_normalizer:
                # fit on *this* df slice (use only for train env!)
                mu = X_raw.mean()
                sd = X_raw.std().replace(0, 1)
                X = ((X_raw - mu) / sd).astype(np.float32).values
                norm_stats_used = {"mu": mu.to_dict(), "sd": sd.to_dict()}
            else:
                # fallback: compute on this slice (ok for quick experiments)
                mu = X_raw.mean()
                sd = X_raw.std().replace(0, 1)
                X = ((X_raw - mu) / sd).astype(np.float32).values
                norm_stats_used = {"mu": mu.to_dict(), "sd": sd.to_dict()}
        else:
            X = X_raw.values
            norm_stats_used = None

        # Spaces
        obs_dim = X.shape[1]
        act_dim = 1

        self.observation_space = Space(low=[-np.inf]*obs_dim,
                                       high=[ np.inf]*obs_dim,
                                       shape=[obs_dim])
        if self.use_delta_actions:
            self.action_space = Space(low=[-self.delta_max], high=[self.delta_max], shape=[act_dim])
        else:
            self.action_space = Space(low=[bid_low], high=[bid_high], shape=[act_dim])

        # Bind to self (what step/reset use)
        self.df   = df
        self.X    = X
        self.DAM  = DAM
        self.BM   = BM
        self.side = SIDE
        self.q    = Q
        self.REF  = REF

        self.feature_cols  = feats
        self.price_col     = price_col
        self.bm_price_col  = bm_price_col
        self.side_col      = side_col
        self.volume_col    = use_vol_col
        self.ref_price_col = ref_price_col

        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.episode_len  = int(episode_len)
        self.rng          = np.random.default_rng(seed)
        self.per_mwh_fee  = float(per_mwh_fee)
        self.fixed_fee    = float(fixed_fee)
        self.reward_scale = float(reward_scale)

        # Expose stats for reuse on val/test
        self.norm_stats_used = norm_stats_used

        # Internal pointers
        self.t0 = 0
        self.t  = 0





    def reset(self):
        """
        Start a new episode at a random time index, keeping a full episode_len window available.
        Returns the first observation s_t.
        """
        max_start = len(self.X) - self.episode_len - 2
        if max_start < 1:
            raise RuntimeError("Not enough rows to run an episode with the current episode_len.")
        self.t0 = int(self.rng.integers(0, max_start))
        self.t = self.t0
        return self.X[self.t]

    def _clears_in_DAM(self, limit_price: float, dam_price: float, side: int) -> bool:
        """
        Clearing rule (deterministic fill decision in DAM):
          • BUY  (side=1): clears if limit >= DAM  (you’re willing to pay up to your bid)
          • SELL (side=0): clears if limit <= DAM  (you’re willing to sell down to your offer)
        """
        return (limit_price >= dam_price) if side == 1 else (limit_price <= dam_price)

    def _pnl(self, cleared: bool, dam: float, bm: float, q: float, side: int) -> float:
        """
        Realized P&L if cleared, else 0; scaled by reward_scale.
        BUY  (side=1):  (BM - DAM) * q
        SELL (side=0):  (DAM - BM) * q  ==  (-(BM - DAM)) * q
        Unified with sign = (2*side - 1) ∈ {+1, -1}.
        Fees are charged only when there is a fill.
        """
        if not cleared:
            return 0.0
        # sign = +1 for buy, -1 for sell in (BM - DAM)
        sign = (2*side - 1)
        gross = sign * (bm - dam) * q
        costs = self.per_mwh_fee * q + self.fixed_fee
        return self.reward_scale * (gross - costs)

    def step(self, action):
        """
        One step:
          - interpret action (δ or absolute limit),
          - decide fill, compute reward,
          - advance time and return (next_obs, reward, done, info).
        """
        dam = float(self.DAM[self.t])
        bm  = float(self.BM[self.t])
        s   = int(self.side[self.t])
        q   = float(self.q[self.t])

        if self.use_delta_actions:
            ref = float(self.REF[self.t])  # known pre-gate
            delta = float(np.clip(action[0], self.action_space.low[0], self.action_space.high[0]))
            limit = ref + delta
            delta_out = limit - ref
        else:
            ref = float("nan")
            delta = float("nan")
            limit = float(np.clip(action[0], self.action_space.low[0], self.action_space.high[0]))
            delta_out = float("nan")

        cleared = self._clears_in_DAM(limit, dam, s)
        reward = self._pnl(cleared, dam, bm, q, s)

        self.t += 1
        done = (self.t >= self.t0 + self.episode_len)
        next_obs = self.X[self.t] if not done else self.X[self.t - 1]

        info = {
            "DeliveryPeriod": self.df.at[self.t - 1, "DeliveryPeriod"],
            "DAM": dam, "BM": bm, "ref": ref,
            "side": s, "q": q,
            "limit": limit, "delta": delta_out,
            "cleared": bool(cleared),
            "reward": reward,
        }
        return next_obs, reward, done, info

# Expose for trainers
    @property
    def reward_scale(self) -> float:
        return self._reward_scale

    @reward_scale.setter
    def reward_scale(self, v: float):
        self._reward_scale = float(v)