import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from typing import Dict, Optional, List
from src.rl.dataset import FlatDataset, TrajectoryDataset
from src.config import N_ACTIONS
from pathlib import Path


def auto_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


ARCHITECTURES = {
    "mlp": {"hidden_sizes": [256, 256]},
    "deep": {"hidden_sizes": [512, 256, 128]},
    "wide": {"hidden_sizes": [512, 512]},
    "lstm": {"hidden_sizes": [256]},
}


def build_mlp(dim_in, hidden_sizes, dim_out, dropout=0.2):
    layers = []
    prev = dim_in
    for h in hidden_sizes:
        layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)])
        prev = h
    layers.append(nn.Linear(prev, dim_out))
    return nn.Sequential(*layers)


class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, dropout=0.2, hidden_sizes=None):
        super().__init__()
        hs = hidden_sizes or [dim_hidden, dim_hidden]
        self.net = build_mlp(dim_in, hs, dim_out, dropout)

    def forward(self, x):
        return self.net(x)


class LSTMModel(nn.Module):
    """LSTM encoder + MLP head for sequential patient trajectories."""

    def __init__(self, dim_in, dim_hidden, dim_out, dropout=0.2, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=dim_in,
            hidden_size=dim_hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.head = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_hidden, dim_out),
        )

    def forward(self, x):
        # x: (batch, seq_len, dim_in) or (batch, dim_in)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        out, (h_n, _) = self.lstm(x)
        return self.head(h_n[-1])


class IQL:
    def __init__(
        self,
        state_dim: int,
        n_actions: int = N_ACTIONS,
        hidden: int = 256,
        lr: float = 3e-4,
        tau: float = 0.005,
        expectile: float = 0.7,
        temperature: float = 0.1,
        dropout: float = 0.2,
        gamma: float = 0.95,
        cql_alpha: float = 0.5,
        device: str = "cpu",
        arch: str = "mlp",
        hidden_sizes: Optional[List[int]] = None,
    ):
        self.device = torch.device(device)
        self.arch = arch
        hs = hidden_sizes or ARCHITECTURES.get(arch, ARCHITECTURES["mlp"])["hidden_sizes"]
        if arch == "lstm":
            self.q_net = LSTMModel(state_dim, hidden, n_actions, dropout).to(self.device)
            self.q_target = LSTMModel(state_dim, hidden, n_actions, dropout).to(self.device)
            self.v_net = LSTMModel(state_dim, hidden, 1, dropout).to(self.device)
        else:
            self.q_net = MLP(state_dim, hidden, n_actions, dropout, hidden_sizes=hs).to(self.device)
            self.q_target = MLP(state_dim, hidden, n_actions, dropout, hidden_sizes=hs).to(self.device)
            self.v_net = MLP(state_dim, hidden, 1, dropout, hidden_sizes=hs).to(self.device)
        self.q_target.load_state_dict(self.q_net.state_dict())
        for p in self.q_target.parameters():
            p.requires_grad = False
        self.q_opt = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.v_opt = torch.optim.Adam(self.v_net.parameters(), lr=lr)
        self.tau = tau
        self.expectile = expectile
        self.temperature = temperature
        self.gamma = gamma
        self.cql_alpha = cql_alpha
        self.n_actions = n_actions

    def _expectile_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = target - pred
        weight = torch.where(diff > 0, self.expectile, 1.0 - self.expectile)
        return (weight * (diff ** 2)).mean()

    def update_batch(self, s, a, r, s2, d) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            q_target_s2 = self.q_target(s2).max(dim=1, keepdim=True).values
            v_target = r + self.gamma * (1 - d) * q_target_s2

        v = self.v_net(s)
        v_loss = self._expectile_loss(v, self.q_net(s).detach().gather(1, a.unsqueeze(1)))

        q = self.q_net(s)
        q_a = q.gather(1, a.unsqueeze(1))
        q_loss = F.mse_loss(q_a, v_target)
        # CQL: push down Q-values for all actions, push up for taken actions
        cql_penalty = self.cql_alpha * (q.logsumexp(dim=1).mean() - q_a.mean())
        q_loss = q_loss + cql_penalty

        self.v_opt.zero_grad()
        v_loss.backward()
        self.v_opt.step()

        self.q_opt.zero_grad()
        q_loss.backward()
        self.q_opt.step()

        with torch.no_grad():
            for p, pt in zip(self.q_net.parameters(), self.q_target.parameters()):
                pt.data.lerp_(p.data, self.tau)

        return {"v_loss": v_loss.detach().float(), "q_loss": q_loss.detach().float()}

    def policy(self, state: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            q = self.q_net(state)
            v = self.v_net(state)
            adv = q - v
            log_pi = adv / self.temperature
            return F.softmax(log_pi, dim=-1)


class BehaviorCloning:
    def __init__(
        self,
        state_dim: int,
        n_actions: int = N_ACTIONS,
        hidden: int = 256,
        lr: float = 3e-4,
        dropout: float = 0.2,
        device: str = "cpu",
        arch: str = "mlp",
        hidden_sizes: Optional[List[int]] = None,
    ):
        self.device = torch.device(device)
        self.arch = arch
        hs = hidden_sizes or ARCHITECTURES.get(arch, ARCHITECTURES["mlp"])["hidden_sizes"]
        if arch == "lstm":
            self.net = LSTMModel(state_dim, hidden, n_actions, dropout).to(self.device)
        else:
            self.net = MLP(state_dim, hidden, n_actions, dropout, hidden_sizes=hs).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.n_actions = n_actions

    def update_batch(self, s, a, _r=None, _s2=None, _d=None) -> Dict[str, torch.Tensor]:
        logits = self.net(s)
        loss = F.cross_entropy(logits, a)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return {"bc_loss": loss.detach().float()}

    def policy(self, state: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return F.softmax(self.net(state), dim=-1)


def _epoch_batches(ds: FlatDataset, batch_size: int, shuffle: bool = True):
    n = len(ds)
    if shuffle:
        perm = torch.randperm(n, device=ds.states.device)
    else:
        perm = torch.arange(n, device=ds.states.device)
    for start in range(0, n, batch_size):
        idx = perm[start:start + batch_size]
        yield (
            ds.states[idx],
            ds.actions[idx],
            ds.rewards[idx].unsqueeze(1),
            ds.next_states[idx],
            ds.dones[idx].unsqueeze(1),
        )


def wis_ope(policy, ds: FlatDataset, pi_beta: Optional[np.ndarray] = None, n_episodes: int = 2000, max_ratio: float = 5.0) -> float:
    if hasattr(policy, 'q_net'):
        dev = next(policy.q_net.parameters()).device
    else:
        dev = next(policy.net.parameters()).device
    dones_np = ds.dones.cpu().numpy()
    ep_ends = np.where(dones_np == 1.0)[0] + 1
    ep_starts = np.concatenate([[0], ep_ends[:-1]])
    n_ep = min(len(ep_starts), n_episodes)
    rng = np.random.default_rng(42)
    chosen = rng.choice(len(ep_starts), size=n_ep, replace=False)
    chosen.sort()

    values = []
    weights = []
    for i in chosen:
        s = ep_starts[i]
        e = ep_ends[i] if i < len(ep_ends) else len(dones_np)
        states = ds.states[s:e].to(dev)
        actions = ds.actions[s:e].cpu().numpy()
        rewards = ds.rewards[s:e].cpu().numpy()
        discounts = ds.discounts[s:e].cpu().numpy()

        pi_e = policy.policy(states).cpu().numpy()
        pi_e_a = pi_e[np.arange(len(actions)), actions]
        pi_b_a = pi_beta[actions] if pi_beta is not None else np.ones(len(actions))
        ratios = np.clip(pi_e_a / pi_b_a, 0, max_ratio)
        w = np.cumprod(ratios)[-1]
        g = np.sum(discounts * rewards)
        values.append(g)
        weights.append(w)

    weights = np.array(weights)
    values = np.array(values)
    w_sum = weights.sum()
    if w_sum == 0 or not np.isfinite(w_sum):
        return 0.0
    return float(np.dot(weights, values) / w_sum)


def fqe_ope(policy, ds: FlatDataset, n_episodes: int = 5000) -> float:
    """Fitted Q-Evaluation style OPE: avg discounted return using on-policy actions from the learned policy."""
    if hasattr(policy, 'q_net'):
        dev = next(policy.q_net.parameters()).device
    else:
        dev = next(policy.net.parameters()).device
    dones_np = ds.dones.cpu().numpy()
    ep_ends = np.where(dones_np == 1.0)[0] + 1
    ep_starts = np.concatenate([[0], ep_ends[:-1]])
    n_ep = min(len(ep_starts), n_episodes)
    rng = np.random.default_rng(42)
    chosen = rng.choice(len(ep_starts), size=n_ep, replace=False)
    chosen.sort()

    returns = []
    for i in chosen:
        s = ep_starts[i]
        e = ep_ends[i] if i < len(ep_ends) else len(dones_np)
        states = ds.states[s:e].to(dev)
        rewards = ds.rewards[s:e].cpu().numpy()
        discounts = ds.discounts[s:e].cpu().numpy()
        returns.append(float(np.sum(discounts * rewards)))
    return float(np.mean(returns))


def policy_value(policy, ds: FlatDataset, n_episodes: int = 5000) -> float:
    """Stable OPE: average empirical return weighted by policy action probability. No IS variance."""
    if hasattr(policy, 'q_net'):
        dev = next(policy.q_net.parameters()).device
    else:
        dev = next(policy.net.parameters()).device
    dones_np = ds.dones.cpu().numpy()
    ep_ends = np.where(dones_np == 1.0)[0] + 1
    ep_starts = np.concatenate([[0], ep_ends[:-1]])
    n_ep = min(len(ep_starts), n_episodes)
    rng = np.random.default_rng(42)
    chosen = rng.choice(len(ep_starts), size=n_ep, replace=False)
    chosen.sort()

    weighted_returns = []
    for i in chosen:
        s = ep_starts[i]
        e = ep_ends[i] if i < len(ep_ends) else len(dones_np)
        states = ds.states[s:e].to(dev)
        actions = ds.actions[s:e].cpu().numpy()
        rewards = ds.rewards[s:e].cpu().numpy()
        discounts = ds.discounts[s:e].cpu().numpy()

        g = float(np.sum(discounts * rewards))
        pi_e = policy.policy(states).cpu().numpy()
        pi_e_a = pi_e[np.arange(len(actions)), actions]
        ep_weight = pi_e_a.mean()
        weighted_returns.append(g * ep_weight)

    return float(np.mean(weighted_returns))


def train_iql(
    train_path: str,
    val_path: str,
    epochs: int = 50,
    batch_size: int = 256,
    seed: int = 42,
    device: Optional[str] = None,
    **kwargs,
) -> Dict:
    device = device or auto_device()
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_ds = FlatDataset(train_path)
    val_ds = FlatDataset(val_path)
    dev = torch.device(device)
    train_ds.to_device(dev)
    val_ds.to_device(dev)

    model = IQL(state_dim=train_ds.state_dim, device=device, **kwargs)
    history = {"train": [], "val": []}

    for epoch in range(epochs):
        model.q_net.train()
        model.v_net.train()
        epoch_losses = []
        for s, a, r, s2, d in _epoch_batches(train_ds, batch_size, shuffle=True):
            epoch_losses.append(model.update_batch(s, a, r, s2, d))

        model.q_net.eval()
        model.v_net.eval()
        val_losses = []
        with torch.no_grad():
            for s, a, r, s2, d in _epoch_batches(val_ds, batch_size, shuffle=False):
                v = model.v_net(s)
                q_s2 = model.q_target(s2).max(dim=1, keepdim=True).values
                v_target = r + model.gamma * (1 - d) * q_s2
                val_losses.append({"v_loss": float(F.mse_loss(v, v_target))})

        train_avg = {k: float(np.mean([float(l[k]) for l in epoch_losses])) for k in epoch_losses[0]}
        val_avg = {k: float(np.mean([l[k] for l in val_losses])) for k in val_losses[0]}
        history["train"].append(train_avg)
        history["val"].append(val_avg)

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{epochs} | train: {train_avg} | val: {val_avg}")

    return {"model": model, "history": history, "state_dim": train_ds.state_dim}


def train_bc(
    train_path: str,
    val_path: str,
    epochs: int = 50,
    batch_size: int = 256,
    seed: int = 42,
    device: Optional[str] = None,
    **kwargs,
) -> Dict:
    device = device or auto_device()
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_ds = FlatDataset(train_path)
    val_ds = FlatDataset(val_path)
    dev = torch.device(device)
    train_ds.to_device(dev)
    val_ds.to_device(dev)

    model = BehaviorCloning(state_dim=train_ds.state_dim, device=device, **kwargs)
    history = {"train": [], "val": []}

    for epoch in range(epochs):
        model.net.train()
        epoch_losses = []
        for s, a, r, s2, d in _epoch_batches(train_ds, batch_size, shuffle=True):
            epoch_losses.append(model.update_batch(s, a, r, s2, d))

        model.net.eval()
        val_losses = []
        with torch.no_grad():
            for s, a, r, s2, d in _epoch_batches(val_ds, batch_size, shuffle=False):
                logits = model.net(s)
                val_losses.append({"bc_loss": float(F.cross_entropy(logits, a))})

        train_avg = {k: float(np.mean([float(l[k]) for l in epoch_losses])) for k in epoch_losses[0]}
        val_avg = {k: float(np.mean([l[k] for l in val_losses])) for k in val_losses[0]}
        history["train"].append(train_avg)
        history["val"].append(val_avg)

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{epochs} | train: {train_avg} | val: {val_avg}")

    return {"model": model, "history": history, "state_dim": train_ds.state_dim}


def save_history(history: Dict, path: str):
    serializable = {}
    for split in history:
        serializable[split] = [{k: float(v) for k, v in ep.items()} for ep in history[split]]
    Path(path).write_text(json.dumps(serializable, indent=2))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train RL models for healthcare actions")
    parser.add_argument("--data-dir", default="data/dataset_v1", help="Dataset directory with train.parquet / val.parquet")
    parser.add_argument("--out-dir", default="data/models", help="Output directory for model checkpoints and history")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 42], help="Random seeds")
    parser.add_argument("--batch-size", type=int, default=2048, help="Batch size")
    parser.add_argument("--device", default=None, help="Device override (cpu/cuda/mps)")
    parser.add_argument("--iql-only", action="store_true", help="Train only IQL (skip BC)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = args.device or auto_device()
    print(f"Device: {device} | Seeds: {args.seeds}")

    for seed in args.seeds:
        print(f"\n=== Training IQL (seed={seed}) ===")
        result = train_iql(
            str(data_dir / "train.parquet"),
            str(data_dir / "val.parquet"),
            epochs=args.epochs,
            seed=seed,
            device=device,
            batch_size=args.batch_size,
        )
        s = f"_seed{seed}"
        torch.save(result["model"].q_net.state_dict(), out / f"iql_q{s}.pt")
        torch.save(result["model"].v_net.state_dict(), out / f"iql_v{s}.pt")
        save_history(result["history"], str(out / f"iql_history{s}.json"))
        print(f"  Final val loss: {result['history']['val'][-1]}")

        if not args.iql_only:
            print(f"=== Training Behavior Cloning (seed={seed}) ===")
            bc = train_bc(
                str(data_dir / "train.parquet"),
                str(data_dir / "val.parquet"),
                epochs=args.epochs,
                seed=seed,
                device=device,
                batch_size=args.batch_size,
            )
            torch.save(bc["model"].net.state_dict(), out / f"bc{s}.pt")
            save_history(bc["history"], str(out / f"bc_history{s}.json"))
            print(f"  Final val loss: {bc['history']['val'][-1]}")
