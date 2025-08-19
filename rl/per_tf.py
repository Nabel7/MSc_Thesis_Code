# rl/per_tf.py
from typing import List, Tuple
import numpy as np

# -------- SumTree (no knowledge of max_priority/min_p) --------
class SumTree:
    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        # 1-indexed binary tree stored as array; leaves are [capacity .. 2*capacity-1]
        self.tree = np.zeros(2 * self.capacity, dtype=np.float32)
        self.data = [None] * self.capacity
        self.write = 0
        self.n_entries = 0
        self.eps_leaf = 1e-6  # floor for any leaf priority written to the tree

    def _propagate(self, idx: int, change: float):
        parent = idx // 2
        self.tree[parent] += change
        if parent > 1:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float) -> int:
        left = 2 * idx
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self) -> float:
        return float(self.tree[1])

    def add(self, p: float, data):
        # write at current pointer's leaf
        idx = self.write + self.capacity
        self.data[self.write] = data
        # enforce a minimal positive priority at the leaf
        self.update(idx, max(float(p), self.eps_leaf))
        # advance pointer
        self.write = (self.write + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, idx: int, p: float):
        p = float(p)
        change = p - float(self.tree[idx])
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s: float):
        idx = self._retrieve(1, float(s))
        dataIdx = idx - self.capacity
        return (idx, float(self.tree[idx]), self.data[dataIdx])


# -------- Prioritized Replay Buffer (owns max_priority/min_p) --------
class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, alpha: float = 0.6, eps: float = 1e-5):
        self.alpha = float(alpha)
        self.eps = float(eps)           # small additive epsilon in priority formula
        self.min_p = 1e-6               # floor for stored priorities (at leaves)
        self.tree = SumTree(capacity)
        self.max_priority = 1.0         # track the max TD-error seen (for new items)

    def __len__(self):
        return self.tree.n_entries

    def add(self, transition):
        # New items receive current max priority => guaranteed to be sampled soon
        p = (self.max_priority + self.eps) ** self.alpha
        self.tree.add(p, transition)

    def sample(self, batch_size: int, beta: float = 0.4):
        beta = float(beta)
        total = self.tree.total()
        # should never be zero thanks to min_p, but guard anyway
        if total <= 0.0 or not np.isfinite(total):
            # fallback to uniform sampling over filled slots
            n = max(1, self.tree.n_entries)
            idxs = []
            batch = []
            priorities = []
            for i in np.random.randint(0, n, size=batch_size):
                leaf = i % n
                idx = leaf + self.tree.capacity
                idxs.append(idx)
                priorities.append(self.min_p)
                batch.append(self.tree.data[leaf])
            probs = np.full(batch_size, 1.0 / n, dtype=np.float32)
        else:
            segment = total / batch_size
            idxs, batch, priorities = [], [], []
            for i in range(batch_size):
                s = np.random.uniform(segment * i, segment * (i + 1))
                idx, p, data = self.tree.get(s)
                idxs.append(idx)
                priorities.append(max(p, self.min_p))
                batch.append(data)
            probs = np.array(priorities, dtype=np.float32) / max(total, 1e-8)

        # importance-sampling weights
        n_entries = max(1, self.tree.n_entries)
        weights = (n_entries * probs) ** (-beta)
        # normalize weights robustly
        w_max = np.max(weights)
        if not np.isfinite(w_max) or w_max <= 0:
            weights = np.ones_like(weights, dtype=np.float32)
        else:
            weights = (weights / w_max).astype(np.float32)

        return idxs, batch, weights

    def update_priorities(self, idxs: List[int], priorities):
        # priorities are raw |TD error| (or similar); apply eps & alpha, clamp floor
        for idx, p in zip(idxs, priorities):
            p = float(p)
            if not np.isfinite(p):
                p = self.min_p
            leaf_p = max(p + self.eps, self.min_p)
            self.max_priority = max(self.max_priority, leaf_p)
            self.tree.update(int(idx), leaf_p ** self.alpha)
