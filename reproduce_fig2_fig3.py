"""Reproduce Figures 2 and 3 from Sussillo & Barak (2013) with PyTorch.

The script trains a 3-bit flip-flop echo-state RNN with FORCE/RLS-style
updates to the readout weights, then searches fixed points in the zero-input
autonomous dynamics and plots low-dimensional phase-space summaries.

The defaults are chosen for a CPU-friendly first pass. Increase ``--n`` and
``--train-steps`` for a closer, slower run.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


@dataclass
class Config:
    seed: int = 3
    n: int = 256
    g: float = 1.5
    dt: float = 0.1
    train_steps: int = 9000
    test_steps: int = 900
    pulse_width: int = 6
    min_interval: int = 22
    max_interval: int = 55
    input_scale: float = 1.0
    feedback_scale: float = 1.0
    feedback_during_training: str = "output"
    rls_alpha: float = 1.0
    rls_every: int = 2
    washout_steps: int = 150
    settle_steps: int = 260
    transition_relax_steps: int = 160
    fixed_point_ics: int = 360
    fixed_point_steps: int = 1800
    fixed_point_lr: float = 0.035
    fixed_point_q_thresh: float = 1e-6
    cluster_distance: float = 0.65
    unstable_tol: float = 1e-3
    out_dir: str = "outputs"
    device: str = "cpu"


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_pulse_task(
    steps: int,
    pulse_width: int,
    min_interval: int,
    max_interval: int,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate sparse 3-channel pulses and persistent +/-1 memory targets."""
    u = np.zeros((steps, 3), dtype=np.float32)
    y = np.zeros((steps, 3), dtype=np.float32)
    state = np.zeros(3, dtype=np.float32)
    starts: dict[int, tuple[int, float]] = {}

    cursor = 0
    for bit in rng.permutation(3):
        sign = float(rng.choice([-1.0, 1.0]))
        starts[cursor] = (int(bit), sign)
        u[cursor : cursor + pulse_width, bit] = sign
        cursor += pulse_width + 2

    next_pulse = cursor + int(rng.integers(min_interval, max_interval + 1))
    while next_pulse + pulse_width < steps:
        bit = int(rng.integers(0, 3))
        sign = float(rng.choice([-1.0, 1.0]))
        starts[next_pulse] = (bit, sign)
        u[next_pulse : next_pulse + pulse_width, bit] = sign
        next_pulse += pulse_width + int(rng.integers(min_interval, max_interval + 1))

    for t in range(steps):
        if t in starts:
            bit, sign = starts[t]
            state[bit] = sign
        y[t] = state
    return torch.from_numpy(u), torch.from_numpy(y)


def init_network(cfg: Config) -> dict[str, torch.Tensor]:
    n = cfg.n
    dev = torch.device(cfg.device)
    j = cfg.g * torch.randn(n, n, device=dev) / math.sqrt(n)
    b = cfg.input_scale * torch.randn(n, 3, device=dev) / math.sqrt(3)
    wfb = cfg.feedback_scale * torch.randn(n, 3, device=dev) / math.sqrt(3)
    wout = torch.zeros(3, n, device=dev)
    return {"j": j, "b": b, "wfb": wfb, "wout": wout}


def step_rnn(
    x: torch.Tensor,
    u_t: torch.Tensor,
    net: dict[str, torch.Tensor],
    dt: float,
    wout_override: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    wout = net["wout"] if wout_override is None else wout_override
    r = torch.tanh(x)
    z = wout @ r
    dx = -x + net["j"] @ r + net["wfb"] @ z + net["b"] @ u_t
    return x + dt * dx, r, z


def train_force(
    cfg: Config,
    net: dict[str, torch.Tensor],
    u: torch.Tensor,
    target: torch.Tensor,
) -> list[float]:
    dev = torch.device(cfg.device)
    u = u.to(dev)
    target = target.to(dev)
    n = cfg.n
    x = torch.zeros(n, device=dev)
    p = torch.eye(n, device=dev) / cfg.rls_alpha
    losses: list[float] = []
    if cfg.feedback_during_training not in {"target", "output"}:
        raise ValueError("--feedback-during-training must be 'target' or 'output'")

    for t in range(cfg.train_steps):
        r = torch.tanh(x)
        z = net["wout"] @ r
        feedback = target[t] if cfg.feedback_during_training == "target" else z
        dx = -x + net["j"] @ r + net["wfb"] @ feedback + net["b"] @ u[t]
        x = x + cfg.dt * dx
        err = target[t] - z
        if t >= cfg.washout_steps and t % cfg.rls_every == 0:
            pr = p @ r
            denom = 1.0 + torch.dot(r, pr)
            k = pr / denom
            net["wout"] += err[:, None] @ k[None, :]
            p -= k[:, None] @ pr[None, :]
        if t % 100 == 0:
            losses.append(float(torch.mean((z - target[t]) ** 2).detach().cpu()))
    return losses


@torch.no_grad()
def simulate(
    cfg: Config,
    net: dict[str, torch.Tensor],
    u: torch.Tensor,
    x0: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dev = torch.device(cfg.device)
    u = u.to(dev)
    x = torch.zeros(cfg.n, device=dev) if x0 is None else x0.clone().to(dev)
    xs, zs = [], []
    for t in range(u.shape[0]):
        x, _, z = step_rnn(x, u[t], net, cfg.dt)
        xs.append(x.detach().cpu())
        zs.append(z.detach().cpu())
    return torch.stack(xs), torch.stack(zs), u.detach().cpu()


@torch.no_grad()
def settle_to_memory(cfg: Config, net: dict[str, torch.Tensor], memory: np.ndarray) -> torch.Tensor:
    dev = torch.device(cfg.device)
    x = torch.zeros(cfg.n, device=dev)
    for bit, sign in enumerate(memory):
        u = torch.zeros(1, cfg.pulse_width, 3, device=dev)
        u[0, :, bit] = float(sign)
        for t in range(cfg.pulse_width):
            x, _, _ = step_rnn(x, u[0, t], net, cfg.dt)
        for _ in range(cfg.settle_steps // 3):
            x, _, _ = step_rnn(x, torch.zeros(3, device=dev), net, cfg.dt)
    for _ in range(cfg.settle_steps):
        x, _, _ = step_rnn(x, torch.zeros(3, device=dev), net, cfg.dt)
    return x.detach().cpu()


@torch.no_grad()
def make_transition_trajectories(
    cfg: Config,
    net: dict[str, torch.Tensor],
    memory_x: dict[tuple[int, int, int], torch.Tensor],
    amplitudes: list[float] | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    dev = torch.device(cfg.device)
    transitions: list[dict[str, object]] = []
    detailed: list[dict[str, object]] = []

    for mem, x0_cpu in memory_x.items():
        mem_arr = np.array(mem, dtype=np.float32)
        for bit in range(3):
            target_sign = -mem_arr[bit]
            u = torch.zeros(cfg.pulse_width + cfg.transition_relax_steps, 3, device=dev)
            u[: cfg.pulse_width, bit] = float(target_sign)
            xs, _, uu = simulate(cfg, net, u.cpu(), x0_cpu)
            new_mem = mem_arr.copy()
            new_mem[bit] = target_sign
            transitions.append(
                {
                    "from": tuple(int(v) for v in mem_arr),
                    "to": tuple(int(v) for v in new_mem),
                    "bit": bit,
                    "xs": xs,
                    "u": uu,
                }
            )

    if amplitudes is None:
        amplitudes = [0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0]
    source = (-1, -1, -1)
    x0_cpu = memory_x[source]
    for amp in amplitudes:
        u = torch.zeros(cfg.pulse_width + cfg.transition_relax_steps, 3, device=dev)
        u[: cfg.pulse_width, 0] = amp
        xs, _, uu = simulate(cfg, net, u.cpu(), x0_cpu)
        detailed.append({"amp": amp, "xs": xs, "u": uu})

    return transitions, detailed


def autonomous_velocity(x: torch.Tensor, jeff: torch.Tensor) -> torch.Tensor:
    return -x + torch.tanh(x) @ jeff.T


def find_fixed_points(
    cfg: Config,
    net: dict[str, torch.Tensor],
    initial_states: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    dev = torch.device(cfg.device)
    j = net["j"].to(dev)
    wfb = net["wfb"].to(dev)
    wout = net["wout"].to(dev)
    jeff = j + wfb @ wout

    if initial_states.shape[0] > cfg.fixed_point_ics:
        idx = torch.randperm(initial_states.shape[0])[: cfg.fixed_point_ics]
        initial_states = initial_states[idx]
    x = initial_states.to(dev).clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([x], lr=cfg.fixed_point_lr)
    best_x = x.detach().clone()
    best_q = torch.full((x.shape[0],), float("inf"), device=dev)

    for step in range(cfg.fixed_point_steps):
        opt.zero_grad(set_to_none=True)
        f = autonomous_velocity(x, jeff)
        q = 0.5 * torch.mean(f * f, dim=1)
        loss = q.mean()
        loss.backward()
        opt.step()
        with torch.no_grad():
            improved = q < best_q
            best_q[improved] = q[improved]
            best_x[improved] = x.detach()[improved]
        if step in (800, 1300):
            for group in opt.param_groups:
                group["lr"] *= 0.35

    with torch.no_grad():
        f = autonomous_velocity(best_x, jeff)
        q = 0.5 * torch.mean(f * f, dim=1)
    return best_x.detach().cpu(), q.detach().cpu()


def cluster_fixed_points(
    xs: torch.Tensor,
    q: torch.Tensor,
    cfg: Config,
) -> tuple[torch.Tensor, torch.Tensor]:
    order = torch.argsort(q)
    centers: list[torch.Tensor] = []
    center_q: list[torch.Tensor] = []
    for idx in order.tolist():
        if float(q[idx]) > cfg.fixed_point_q_thresh:
            continue
        x = xs[idx]
        if not centers:
            centers.append(x)
            center_q.append(q[idx])
            continue
        dists = torch.stack([torch.linalg.norm(x - c) / math.sqrt(cfg.n) for c in centers])
        if float(dists.min()) > cfg.cluster_distance:
            centers.append(x)
            center_q.append(q[idx])
    if not centers:
        return xs[order[:0]], q[order[:0]]
    return torch.stack(centers), torch.stack(center_q)


def pca_fit_transform(data: torch.Tensor, n_components: int = 3) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = data.mean(dim=0)
    centered = data - mean
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    components = vh[:n_components]
    return centered @ components.T, mean, components


def project(x: torch.Tensor, mean: torch.Tensor, components: torch.Tensor) -> torch.Tensor:
    return (x - mean) @ components.T


def classify_fixed_points(
    cfg: Config,
    net: dict[str, torch.Tensor],
    fps: torch.Tensor,
) -> list[dict[str, object]]:
    jeff = (net["j"] + net["wfb"] @ net["wout"]).detach().cpu().numpy()
    wout = net["wout"].detach().cpu().numpy()
    out: list[dict[str, object]] = []
    eye = np.eye(cfg.n, dtype=np.float64)
    for i, fp in enumerate(fps.numpy()):
        r = np.tanh(fp)
        jac = jeff * (1.0 - r * r)[None, :] - eye
        vals, vecs = np.linalg.eig(jac)
        unstable = np.where(vals.real > cfg.unstable_tol)[0]
        top = np.argsort(vals.real)[::-1][: min(4, len(vals))]
        z = wout @ r
        out.append(
            {
                "index": i,
                "output": z.tolist(),
                "memory": np.sign(z).astype(int).tolist(),
                "unstable_count": int(len(unstable)),
                "eig_real_max": float(vals.real.max()),
                "unstable_vectors": vecs[:, unstable[:4]].real.astype(np.float32),
                "top_vectors": vecs[:, top].real.astype(np.float32),
            }
        )
    return out


def fixed_point_json_summary(fp_info: list[dict[str, object]]) -> list[dict[str, object]]:
    """Drop eigenvectors from fixed-point metadata before JSON serialization."""
    keep = []
    for item in fp_info:
        keep.append(
            {
                "index": item["index"],
                "output": item["output"],
                "memory": item["memory"],
                "unstable_count": item["unstable_count"],
                "eig_real_max": item["eig_real_max"],
            }
        )
    return keep


def plot_figure2(
    out_path: Path,
    u: torch.Tensor,
    z: torch.Tensor,
    target: torch.Tensor,
    train_losses: list[float],
) -> None:
    colors = [("#7f1515", "#ff2020"), ("#0f3b1d", "#19a35b"), ("#173a92", "#315fd6")]
    t = np.arange(u.shape[0])
    fig = plt.figure(figsize=(12, 5.2), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0])
    ax = fig.add_subplot(gs[0, 0])
    offsets = np.array([2.7, 0.0, -2.7])
    for k, (dark, light) in enumerate(colors):
        ax.plot(t, u[:, k].numpy() * 0.65 + offsets[k], color=dark, lw=1.4)
        ax.plot(t, z[:, k].numpy() * 0.75 + offsets[k], color=light, lw=1.7)
        ax.plot(t, target[:, k].numpy() * 0.75 + offsets[k], color=light, lw=0.8, alpha=0.25)
        ax.text(-35, offsets[k] + 0.72, "+1", fontsize=9, ha="right", va="center")
        ax.text(-35, offsets[k] - 0.72, "-1", fontsize=9, ha="right", va="center")
    ax.set_xlim(0, len(t) - 1)
    ax.set_ylim(-4.0, 3.8)
    ax.set_yticks([])
    ax.set_xlabel("Time")
    ax.set_title("Sample input/output traces")
    ax.spines[["top", "right", "left"]].set_visible(False)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_aspect("equal")
    theta = np.linspace(0, 2 * np.pi, 120)
    ax2.plot(0.45 * np.cos(theta), 0.45 * np.sin(theta), color="0.15", lw=2)
    rng = np.random.default_rng(4)
    pts = rng.normal(size=(28, 2))
    pts = pts / np.maximum(np.linalg.norm(pts, axis=1, keepdims=True), 1e-6) * rng.uniform(0.05, 0.38, (28, 1))
    ax2.scatter(pts[:, 0], pts[:, 1], s=18, facecolor="#f1dc6b", edgecolor="0.35", zorder=3)
    for _ in range(34):
        a, b = rng.choice(len(pts), size=2, replace=False)
        ax2.plot([pts[a, 0], pts[b, 0]], [pts[a, 1], pts[b, 1]], color="0.65", lw=0.6, alpha=0.55)
    for y0 in [-0.22, 0.0, 0.22]:
        ax2.annotate("", xy=(-0.45, y0), xytext=(-0.92, y0), arrowprops=dict(arrowstyle="->", lw=1.1))
        ax2.scatter([-1.0], [y0], s=55, facecolor="#f1dc6b", edgecolor="0.25")
    for y0 in [-0.25, 0.0, 0.25]:
        ax2.annotate("", xy=(0.88, y0), xytext=(0.45, y0), arrowprops=dict(arrowstyle="->", lw=1.1))
        ax2.scatter([1.0], [y0], s=72, facecolor="white", edgecolor="red", lw=1.4)
    ax2.plot([1.08, 1.08, 0.25], [0.25, -0.48, -0.48], color="0.15", lw=1.7)
    ax2.annotate("", xy=(0.25, -0.45), xytext=(0.25, -0.6), arrowprops=dict(arrowstyle="->", lw=1.0))
    ax2.text(0, -0.72, f"FORCE/RLS readout training\nlast trace MSE={train_losses[-1]:.2e}", ha="center", fontsize=9)
    ax2.set_xlim(-1.25, 1.25)
    ax2.set_ylim(-0.95, 0.78)
    ax2.set_title("Echo-state architecture")
    ax2.axis("off")

    fig.suptitle("Figure 2 reproduction: 3-bit flip-flop task", fontsize=14)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_figure3(
    out_path: Path,
    cfg: Config,
    net: dict[str, torch.Tensor],
    transitions: list[dict[str, object]],
    detailed: list[dict[str, object]],
    fps: torch.Tensor,
    fp_info: list[dict[str, object]],
) -> None:
    transition_states = torch.cat([item["xs"][::5] for item in transitions], dim=0)
    pca_data = torch.cat([transition_states, fps], dim=0)
    _, mean, comps = pca_fit_transform(pca_data, 3)
    fp_proj = project(fps, mean, comps).numpy()

    fig = plt.figure(figsize=(13, 6), constrained_layout=True)
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    for item in transitions:
        pts = project(item["xs"], mean, comps).numpy()
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color="#2f61d5", lw=0.65, alpha=0.8)

    stable_idx = [i for i, info in enumerate(fp_info) if info["unstable_count"] == 0]
    saddle_idx = [i for i, info in enumerate(fp_info) if info["unstable_count"] == 1]
    other_idx = [i for i, info in enumerate(fp_info) if info["unstable_count"] > 1]
    if stable_idx:
        ax.scatter(*fp_proj[stable_idx].T, marker="x", s=80, c="black", lw=2.0, label="stable")
    if saddle_idx:
        ax.scatter(*fp_proj[saddle_idx].T, marker="x", s=70, c="#2cbf31", lw=2.0, label="1D saddle")
    if other_idx:
        ax.scatter(*fp_proj[other_idx].T, marker="x", s=70, c="#d752a8", lw=2.0, label="other")

    comp_np = comps.numpy()
    mean_np = mean.numpy()
    for i, info in enumerate(fp_info):
        vecs = info["unstable_vectors"]
        if vecs.shape[1] == 0:
            continue
        for k in range(vecs.shape[1]):
            v = torch.from_numpy(vecs[:, k])
            v = v / (torch.linalg.norm(v) + 1e-8)
            line = torch.stack([fps[i] - 1.2 * v, fps[i] + 1.2 * v])
            p = project(line, mean, comps).numpy()
            ax.plot(p[:, 0], p[:, 1], p[:, 2], color="red", lw=1.5, alpha=0.85)

            if info["unstable_count"] == 1 and k == 0:
                for sign in [-1.0, 1.0]:
                    x0 = fps[i] + sign * 0.18 * v
                    u0 = torch.zeros(cfg.transition_relax_steps, 3)
                    xs, _, _ = simulate(cfg, net, u0, x0)
                    pp = project(xs, mean, comps).numpy()
                    ax.plot(pp[:, 0], pp[:, 1], pp[:, 2], color="red", lw=0.5, alpha=0.65)

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title("Fixed points and 1-bit transitions")
    ax.view_init(elev=18, azim=-62)
    ax.legend(loc="upper left", fontsize=8)

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    for item in detailed:
        pts = project(item["xs"], mean, comps).numpy()
        inp = item["u"][:, 0].numpy()
        color = "#2252c7" if abs(float(item["amp"]) - 1.0) < 1e-6 else "#52c7dc"
        lw = 1.7 if color == "#2252c7" else 0.9
        ax2.plot(pts[:, 2], inp, pts[:, 0], color=color, lw=lw, alpha=0.9)
    ax2.set_xlabel("PC3")
    ax2.set_ylabel("input 1")
    ax2.set_zlabel("PC1")
    ax2.set_title("Input-amplitude perturbation")
    ax2.view_init(elev=20, azim=-48)

    fig.suptitle(
        f"Figure 3 reproduction: {len(fps)} fixed points "
        f"({len(stable_idx)} stable, {len(saddle_idx)} saddles)",
        fontsize=14,
    )
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    for field, value in asdict(Config()).items():
        arg = "--" + field.replace("_", "-")
        if isinstance(value, bool):
            parser.add_argument(arg, action="store_true")
        else:
            parser.add_argument(arg, type=type(value), default=value)
    args = parser.parse_args()
    cfg = Config(**{k: getattr(args, k) for k in asdict(Config()).keys()})
    set_seed(cfg.seed)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(cfg.seed)
    net = init_network(cfg)
    u_train, y_train = make_pulse_task(
        cfg.train_steps,
        cfg.pulse_width,
        cfg.min_interval,
        cfg.max_interval,
        rng,
    )
    losses = train_force(cfg, net, u_train, y_train)

    u_test, y_test = make_pulse_task(
        cfg.test_steps,
        cfg.pulse_width,
        cfg.min_interval,
        cfg.max_interval,
        rng,
    )
    xs_test, z_test, u_test = simulate(cfg, net, u_test)
    test_mse = float(torch.mean((z_test - y_test) ** 2))

    memories = [tuple(int(v) for v in vals) for vals in np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1])).T.reshape(-1, 3)]
    memory_x = {mem: settle_to_memory(cfg, net, np.array(mem, dtype=np.float32)) for mem in memories}
    transitions, detailed = make_transition_trajectories(cfg, net, memory_x)

    transition_states = torch.cat([item["xs"][::4] for item in transitions], dim=0)
    memory_states = torch.stack(list(memory_x.values()))
    midpoint_states = []
    for item in transitions:
        midpoint_states.append(item["xs"][cfg.pulse_width + 10])
        midpoint_states.append(item["xs"][cfg.pulse_width + 35])
    fp_initial = torch.cat([transition_states, memory_states, torch.stack(midpoint_states), xs_test[::8]], dim=0)
    fp_candidates, q_candidates = find_fixed_points(cfg, net, fp_initial)
    fps, fp_q = cluster_fixed_points(fp_candidates, q_candidates, cfg)
    fp_info = classify_fixed_points(cfg, net, fps) if len(fps) else []

    plot_figure2(out_dir / "figure2_reproduction.png", u_test, z_test, y_test, losses)
    if len(fps):
        plot_figure3(out_dir / "figure3_reproduction.png", cfg, net, transitions, detailed, fps, fp_info)

    summary = {
        "config": asdict(cfg),
        "train_loss_samples": losses,
        "test_mse": test_mse,
        "num_fixed_points": int(len(fps)),
        "fixed_point_q": [float(v) for v in fp_q],
        "fixed_points": fixed_point_json_summary(fp_info),
        "outputs": {
            "figure2": str(out_dir / "figure2_reproduction.png"),
            "figure3": str(out_dir / "figure3_reproduction.png") if len(fps) else None,
        },
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    torch.save(
        {
            "config": asdict(cfg),
            "net": {k: v.detach().cpu() for k, v in net.items()},
            "fixed_points": fps,
            "fixed_point_q": fp_q,
        },
        out_dir / "reproduction_state.pt",
    )

    print(json.dumps({k: summary[k] for k in ["test_mse", "num_fixed_points", "outputs"]}, indent=2))


if __name__ == "__main__":
    main()
