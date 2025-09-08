# envs/speculator_env.py
import numpy as np
import pandas as pd
from types import SimpleNamespace
from typing import Optional, Iterable, Dict, Any

class Space(SimpleNamespace):
    def __init__(self, low, high, shape):
        super().__init__(low=np.array(low, dtype=np.float32),
                         high=np.array(high, dtype=np.float32),
                         shape=tuple(shape))

class SpeculatorEnv:
    """
    DAM↔BM arbitrage env with deterministic DAM clearing:
      - BUY (side=1) clears if limit >= DAM
      - SELL(side=0) clears if limit <= DAM

    Reward (scaled):
      net = (2*side-1)*(BM-DAM)*q - 1[clear]*(per_mwh_fee*q + fixed_fee) - lambda*|delta|*q
      reward = reward_scale * net
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
                 reward_scale: float = 0.005,
                 normalize_features: bool = False,    # we do external normalization
                 add_time_features: bool = True,
                 use_delta_actions: bool = True,
                 delta_max: float = 50.0,
                 ref_price_col: str = "-24",
                 delta_reg: float = 0.0):

        # Load
        if df is None:
            if csv_path is None:
                raise ValueError("Provide csv_path or df=.")
            df = pd.read_csv(csv_path)

        if "DeliveryPeriod" not in df.columns:
            raise ValueError("CSV must include 'DeliveryPeriod'.")
        df = df.copy()
        df["DeliveryPeriod"] = pd.to_datetime(df["DeliveryPeriod"], errors="coerce")
        df = df.sort_values("DeliveryPeriod").reset_index(drop=True)

        if exclude_years:
            years = df["DeliveryPeriod"].dt.year
            df = df[~years.isin(list(exclude_years))]
        if (start is not None) or (end is not None):
            m = pd.Series(True, index=df.index)
            if start: m &= (df["DeliveryPeriod"] >= pd.to_datetime(start))
            if end:   m &= (df["DeliveryPeriod"] <= pd.to_datetime(end))
            df = df.loc[m]
        df = df.reset_index(drop=True)

        # features (+ optional time features)
        feats = list(feature_cols)
        if add_time_features:
            hr  = df["DeliveryPeriod"].dt.hour.values.astype(np.float32)
            dow = df["DeliveryPeriod"].dt.dayofweek.values.astype(np.float32)
            df["sin_hour"] = np.sin(2*np.pi*hr/24.0).astype(np.float32)
            df["cos_hour"] = np.cos(2*np.pi*hr/24.0).astype(np.float32)
            df["sin_dow"]  = np.sin(2*np.pi*dow/7.0).astype(np.float32)
            df["cos_dow"]  = np.cos(2*np.pi*dow/7.0).astype(np.float32)
            for extra in ["sin_hour","cos_hour","sin_dow","cos_dow"]:
                if extra not in feats: feats.append(extra)

        # guard against leakage
        for leak in [price_col, bm_price_col]:
            if leak in feats:
                raise ValueError(f"'{leak}' in feature_cols — leak through target. Use lags/forecasts/calendar.")

        # action mode
        self.use_delta_actions = bool(use_delta_actions)
        self.delta_max = float(delta_max)
        if self.use_delta_actions:
            if ref_price_col not in df.columns:
                raise ValueError(f"ref_price_col '{ref_price_col}' not found.")
            REF = df[ref_price_col].astype(np.float32).values
        else:
            REF = None

        # volume
        if volume_col and (volume_col in df.columns):
            use_vol_col = volume_col
        elif "ExecQty_MWh" in df.columns:
            df["abs_volume"] = df["ExecQty_MWh"].abs()
            use_vol_col = "abs_volume"
        else:
            df["unit_volume"] = 1.0
            use_vol_col = "unit_volume"

        req = set(feats + [price_col, bm_price_col, side_col, use_vol_col])
        if self.use_delta_actions: req.add(ref_price_col)
        miss = [c for c in req if c not in df.columns]
        if miss: raise ValueError(f"Missing columns: {miss}")

        df = df.dropna(subset=list(req)).reset_index(drop=True)

        # cache arrays
        self.df   = df
        self.X    = df[feats].astype(np.float32).values
        self.DAM  = df[price_col].astype(np.float32).values
        self.BM   = df[bm_price_col].astype(np.float32).values
        self.side = df[side_col].astype(np.int32).values
        self.q    = df[use_vol_col].astype(np.float32).values
        self.REF  = REF

        # spaces
        obs_dim = self.X.shape[1]
        self.observation_space = Space(low=[-np.inf]*obs_dim, high=[np.inf]*obs_dim, shape=[obs_dim])
        if self.use_delta_actions:
            self.action_space = Space(low=[-self.delta_max], high=[self.delta_max], shape=[1])
        else:
            self.action_space = Space(low=[0.0], high=[300.0], shape=[1])

        # params
        self.episode_len  = int(episode_len)
        self.rng          = np.random.default_rng(seed)
        self.per_mwh_fee  = float(per_mwh_fee)
        self.fixed_fee    = float(fixed_fee)
        self.reward_scale = float(reward_scale)
        self.delta_reg    = float(delta_reg)

        self.feature_cols  = feats
        self.ref_price_col = ref_price_col

        self.t0 = 0
        self.t  = 0

    def reset(self):
        max_start = len(self.X) - self.episode_len - 2
        if max_start < 1:
            raise RuntimeError("Not enough rows for the configured episode_len.")
        self.t0 = int(self.rng.integers(0, max_start))
        self.t = self.t0
        return self.X[self.t]

    @staticmethod
    def _clears(limit_price: float, dam_price: float, side: int) -> bool:
        return (limit_price >= dam_price) if side == 1 else (limit_price <= dam_price)

    def step(self, action):
        dam = float(self.DAM[self.t])
        bm  = float(self.BM[self.t])
        s   = int(self.side[self.t])
        q   = float(self.q[self.t])

        if self.use_delta_actions:
            ref   = float(self.REF[self.t])
            delta = float(np.clip(action[0], self.action_space.low[0], self.action_space.high[0]))
            limit = ref + delta
            delta_out = delta
        else:
            ref   = float("nan")
            delta = float("nan")
            limit = float(np.clip(action[0], self.action_space.low[0], self.action_space.high[0]))
            delta_out = float("nan")

        cleared = self._clears(limit, dam, s)

        # gross pnl: + for buy if BM>DAM, - for sell if BM<DAM (sign = 2s-1)
        sign  = (2*s - 1)
        gross = sign * (bm - dam) * q
        fees  = (self.per_mwh_fee * q + self.fixed_fee) if cleared else 0.0
        penalty = self.delta_reg * abs(delta_out) * q

        net = gross - fees - penalty
        reward = self.reward_scale * net

        self.t += 1
        done = (self.t >= self.t0 + self.episode_len)
        next_obs = self.X[self.t] if not done else self.X[self.t - 1]

        info = {
            "DeliveryPeriod": self.df.at[self.t - 1, "DeliveryPeriod"],
            "DAM": dam, "BM": bm, "ref": ref,
            "side": s, "q": q,
            "limit": limit, "delta": delta_out,
            "cleared": bool(cleared),
            "gross": float(gross),
            "fees": float(fees),
            "penalty": float(penalty),
            "reward": float(reward),
        }
        return next_obs, reward, done, info
