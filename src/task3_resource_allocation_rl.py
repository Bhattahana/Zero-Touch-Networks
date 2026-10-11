"""
Task 3 — Zero-touch resource allocation via (tabular) Q-learning.

Frames a classic ZTM RAN-orchestration problem as reinforcement
learning: a controller must repeatedly decide *how* to split a
constrained capacity budget across several concurrently active
sessions, without any human in the loop.

Grounding in real data: at every timestep, each simulated session's
"demand" is the *actual* recorded DL_bitrate of a real trace from the
5G-production-dataset at that point in its playback. This keeps the
workload distribution (bursts, stalls, ramp-ups) real, even though the
pairing of sessions into one "cell" is synthetic (the traces were not
captured simultaneously).

Action space (discrete allocation policies the agent chooses between
each step):
    0 - equal_split        divide capacity evenly across active sessions
    1 - proportional_fair   allocate proportional to each session's demand
    2 - greedy_max          serve the highest-demand session first
    3 - protect_min         serve the lowest-demand session first (fairness)

State: a coarse discretization of (total demand level, demand
imbalance level) over the K active sessions.

Reward: negative normalized unmet demand (i.e. the fraction of
requested throughput that could not be served this step). Q-learning
therefore learns, for each traffic regime, which allocation policy
minimizes QoS violations.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .data_loader import load_dataset

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs"

ACTIONS = ["equal_split", "proportional_fair", "greedy_max", "protect_min"]
N_DEMAND_BUCKETS = 3   # low / medium / high total demand
N_SKEW_BUCKETS = 3     # balanced / mixed / skewed
N_STATES = N_DEMAND_BUCKETS * N_SKEW_BUCKETS
CAPACITY_FRACTION = 0.65  # controller only has 65% of aggregate peak demand


def _bucket(value: float, edges: list[float]) -> int:
    for i, e in enumerate(edges):
        if value <= e:
            return i
    return len(edges)


def build_episodes(df, k_sessions: int = 3, n_episodes: int = 120, min_len: int = 60, seed: int = 0):
    """Groups real session traces into random K-tuples ("cells") to
    play out as RL episodes.
    """
    rng = np.random.default_rng(seed)
    sessions = [g["DL_bitrate"].to_numpy(dtype=np.float64) for _, g in df.groupby("session_id")]
    sessions = [s for s in sessions if len(s) >= min_len]

    episodes = []
    for _ in range(n_episodes):
        chosen = rng.choice(len(sessions), size=k_sessions, replace=False)
        traces = [sessions[i] for i in chosen]
        length = min(len(t) for t in traces)
        length = min(length, 150)  # cap episode length for speed
        episodes.append(np.stack([t[:length] for t in traces], axis=1))  # (T, K)
    return episodes


def _allocate(action_idx: int, demand: np.ndarray, capacity: float) -> np.ndarray:
    """Each policy intentionally behaves differently under scarcity —
    no leftover-capacity redistribution is applied, so the *pattern*
    of who gets served (not just the aggregate total) genuinely
    differs between policies. This is what makes the RL action choice
    matter for the fairness-aware reward below.
    """
    demand = np.maximum(demand, 0.0)
    total = demand.sum()
    if total <= 1e-9:
        return np.zeros_like(demand)

    if ACTIONS[action_idx] == "equal_split":
        # a strict per-session cap; unused headroom from light sessions
        # is NOT handed to heavy ones (simple, but can waste capacity)
        share = capacity / len(demand)
        alloc = np.minimum(demand, share)
    elif ACTIONS[action_idx] == "proportional_fair":
        alloc = demand / total * min(capacity, total)
    elif ACTIONS[action_idx] == "greedy_max":
        # serve the biggest consumers first; small flows starve first
        alloc = np.zeros_like(demand)
        remaining = capacity
        for idx in np.argsort(-demand):
            give = min(demand[idx], remaining)
            alloc[idx] = give
            remaining -= give
            if remaining <= 0:
                break
    elif ACTIONS[action_idx] == "protect_min":
        # serve the smallest consumers first; big flows starve first
        alloc = np.zeros_like(demand)
        remaining = capacity
        for idx in np.argsort(demand):
            give = min(demand[idx], remaining)
            alloc[idx] = give
            remaining -= give
            if remaining <= 0:
                break
    else:
        raise ValueError(action_idx)

    return np.minimum(alloc, demand)


def _state_from_demand(demand: np.ndarray, demand_edges, skew_edges) -> int:
    total = demand.sum()
    skew = demand.std() / (demand.mean() + 1e-9)
    d_bucket = _bucket(total, demand_edges)
    s_bucket = _bucket(skew, skew_edges)
    return d_bucket * N_SKEW_BUCKETS + s_bucket


def _reward(demand: np.ndarray, alloc: np.ndarray) -> float:
    """Fairness-aware QoS reward: penalizes both how much throughput
    is missed on average AND how badly the worst-off session is
    starved, per session (not in aggregate). A policy that fully
    satisfies two sessions while starving a third scores worse here
    than one that spreads the shortfall evenly, even though the
    aggregate unmet bytes can be identical.
    """
    safe_demand = np.maximum(demand, 1e-6)
    unmet_frac = np.clip(demand - alloc, 0, None) / safe_demand
    unmet_frac = np.where(demand > 1e-6, unmet_frac, 0.0)
    if not np.any(demand > 1e-6):
        return 0.0
    mean_unmet = unmet_frac.mean()
    worst_unmet = unmet_frac.max()
    return -float(0.5 * mean_unmet + 0.5 * worst_unmet)


def train_q_learning(
    episodes,
    demand_edges,
    skew_edges,
    n_epochs: int = 25,
    lr: float = 0.2,
    gamma: float = 0.9,
    eps_start: float = 0.3,
    eps_end: float = 0.02,
    seed: int = 0,
):
    rng = np.random.default_rng(seed)
    Q = np.zeros((N_STATES, len(ACTIONS)))
    n_ep = len(episodes)

    reward_history = []
    for epoch in range(n_epochs):
        eps = eps_start + (eps_end - eps_start) * (epoch / max(1, n_epochs - 1))
        epoch_rewards = []
        for ep in episodes:
            T = ep.shape[0]
            capacity = CAPACITY_FRACTION * ep.mean(axis=0).sum()  # constant per episode
            for t in range(T - 1):
                demand = ep[t]
                s = _state_from_demand(demand, demand_edges, skew_edges)

                if rng.random() < eps:
                    a = rng.integers(len(ACTIONS))
                else:
                    a = int(np.argmax(Q[s]))

                alloc = _allocate(a, demand, capacity)
                r = _reward(demand, alloc)

                next_demand = ep[t + 1]
                s_next = _state_from_demand(next_demand, demand_edges, skew_edges)

                Q[s, a] += lr * (r + gamma * Q[s_next].max() - Q[s, a])
                epoch_rewards.append(r)
        reward_history.append(float(np.mean(epoch_rewards)))
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  [q_learning] epoch {epoch + 1}/{n_epochs}  avg_reward={reward_history[-1]:.4f}  eps={eps:.2f}")
    return Q, reward_history


def evaluate_policy(episodes, demand_edges, skew_edges, policy: str, Q=None):
    """policy: 'learned' | 'equal_split' | 'proportional_fair' | 'greedy_max' | 'protect_min' | 'random'"""
    rng = np.random.default_rng(123)
    rewards = []
    for ep in episodes:
        T = ep.shape[0]
        capacity = CAPACITY_FRACTION * ep.mean(axis=0).sum()  # constant per episode
        for t in range(T):
            demand = ep[t]
            if policy == "learned":
                s = _state_from_demand(demand, demand_edges, skew_edges)
                a = int(np.argmax(Q[s]))
            elif policy == "random":
                a = rng.integers(len(ACTIONS))
            else:
                a = ACTIONS.index(policy)
            alloc = _allocate(a, demand, capacity)
            rewards.append(_reward(demand, alloc))
    return float(np.mean(rewards)), float(np.std(rewards))


def run(k_sessions: int = 3, n_episodes: int = 120, max_sessions: int | None = None) -> dict:
    OUT_DIR.mkdir(exist_ok=True)
    print("\n=== TASK 3: Resource allocation (Q-learning) ===")
    raw = load_dataset(max_sessions=max_sessions)

    episodes = build_episodes(raw, k_sessions=k_sessions, n_episodes=n_episodes, seed=0)
    n_train = int(len(episodes) * 0.8)
    train_eps, test_eps = episodes[:n_train], episodes[n_train:]

    # calibrate discretization edges on the training episodes
    all_demands = np.concatenate([ep for ep in train_eps], axis=0)
    totals = all_demands.sum(axis=1)
    skews = all_demands.std(axis=1) / (all_demands.mean(axis=1) + 1e-9)
    demand_edges = list(np.quantile(totals, [1 / 3, 2 / 3]))
    skew_edges = list(np.quantile(skews, [1 / 3, 2 / 3]))

    Q, history = train_q_learning(train_eps, demand_edges, skew_edges, n_epochs=25)

    results = {"reward_curve": history, "actions": ACTIONS, "policies": {}}
    for policy in ["learned", "equal_split", "proportional_fair", "greedy_max", "protect_min", "random"]:
        mean_r, std_r = evaluate_policy(test_eps, demand_edges, skew_edges, policy, Q=Q)
        results["policies"][policy] = {"mean_reward": mean_r, "std_reward": std_r}
        print(f"  [eval] {policy:20s} mean_reward={mean_r:+.4f}  std={std_r:.4f}")

    np.save(OUT_DIR / "task3_q_table.npy", Q)
    (OUT_DIR / "task3_rl_metrics.json").write_text(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    run()
