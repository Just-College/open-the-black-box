"""Run the 3-bit flip-flop RNN teaching demo from Sussillo & Barak (2013).

The script trains a 3-bit flip-flop echo-state RNN with FORCE/RLS-style
updates to the readout weights, then searches fixed points in the zero-input
autonomous dynamics and plots task behavior plus low-dimensional state-space
summaries.

Use ``run_flipflop_demo.py`` as the course entrypoint. This module is
kept as the shared implementation library for training, fixed-point analysis,
cached data generation, and figure rendering helpers.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import MultipleLocator
from scipy.optimize import minimize

NETWORK_DTYPE = torch.float64


@dataclass
class Config:
    seed: int = 8
    n: int = 512
    g: float = 1.3
    dt: float = 0.1
    train_steps: int = 50000
    test_steps: int = 1200
    train_task: str = "mixed"
    structured_fraction: float = 0.7
    pulse_width: int = 6
    min_interval: int = 22
    max_interval: int = 55
    input_scale: float = 1.0
    feedback_scale: float = 1.0
    rls_alpha: float = 1.0
    rls_every: int = 1
    washout_steps: int = 150
    settle_steps: int = 260
    transition_relax_steps: int = 160
    fixed_point_ics: int = 1200
    fixed_point_steps: int = 3000
    fixed_point_lr: float = 0.035
    fixed_point_q_thresh: float = 2.25e-5
    fixed_point_refine: int = 360
    fixed_point_refine_iter: int = 700
    edge_interpolation_points: int = 17
    critical_pulse_ics: bool = True
    critical_bisection_steps: int = 14
    critical_relax_extra: int = 120
    critical_candidate_stride: int = 2
    critical_attractor_q_thresh: float = 1e-5
    cluster_distance: float = 0.4
    unstable_tol: float = 1e-3
    model_dir: str = "model"
    figure_dir: str = "figure"
    device: str = "auto"


PRESETS: dict[str, dict[str, object]] = {
    "full": {},
    "test": {
        "n": 96,
        "train_steps": 1500,
        "test_steps": 320,
        "rls_every": 1,
        "fixed_point_ics": 80,
        "fixed_point_steps": 300,
        "fixed_point_q_thresh": 1e-5,
        "fixed_point_refine": 24,
        "fixed_point_refine_iter": 80,
        "edge_interpolation_points": 5,
        "critical_pulse_ics": False,
        "cluster_distance": 0.55,
    },
}


def make_config(preset: str = "full", **overrides: object) -> Config:
    """Build a run config from a named preset plus explicit Python overrides."""
    if preset not in PRESETS:
        raise ValueError("preset must be one of: full, test")
    values = asdict(Config())
    values.update(PRESETS[preset])
    values.update(overrides)
    return Config(**values)


def config_from_mapping(values: dict[str, object]) -> Config:
    """Load configs saved by older script versions without requiring exact fields."""
    valid = {field.name for field in fields(Config)}
    clean = {key: value for key, value in values.items() if key in valid}
    return Config(**clean)


def save_config_json(cfg: Config, out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=4)


def load_config_json(path: Path) -> Config:
    with path.open("r", encoding="utf-8") as f:
        return config_from_mapping(json.load(f))


def save_model(net: dict[str, torch.Tensor], out_path: Path) -> None:
    torch.save({key: value.detach().cpu() for key, value in net.items()}, out_path)


def load_model(path: Path) -> dict[str, torch.Tensor]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(data, dict) and "net" in data:
        data = data["net"]
    if not isinstance(data, dict):
        raise ValueError(f"{path} does not contain a model state dictionary.")
    return {key: value.detach().cpu() for key, value in data.items()}


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    return dev


def make_pulse_task(
    steps: int,
    pulse_width: int,
    min_interval: int,
    max_interval: int,
    rng: np.random.Generator,
    initial_state: np.ndarray | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate sparse 3-channel pulses and persistent +/-1 memory targets."""
    u = np.zeros((steps, 3), dtype=np.float32)
    y = np.zeros((steps, 3), dtype=np.float32)
    state = (
        np.zeros(3, dtype=np.float32)
        if initial_state is None
        else initial_state.astype(np.float32).copy()
    )
    starts: dict[int, tuple[int, float]] = {}

    cursor = 0
    if initial_state is None:
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


def all_memories() -> list[tuple[int, int, int]]:
    return [
        tuple(int(v) for v in vals)
        for vals in np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1])).T.reshape(-1, 3)
    ]


def make_balanced_transition_task(
    steps: int,
    pulse_width: int,
    min_interval: int,
    max_interval: int,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a balanced sequence that explicitly covers all one-bit transitions."""
    u_rows: list[np.ndarray] = []
    y_rows: list[np.ndarray] = []
    state = np.zeros(3, dtype=np.float32)
    memories = all_memories()
    current = np.array(rng.choice([-1.0, 1.0], size=3), dtype=np.float32)

    def append_steps(n: int, pulse: tuple[int, float] | None = None) -> None:
        nonlocal state
        if pulse is not None:
            bit, sign = pulse
            state[bit] = sign
        for t in range(n):
            u_t = np.zeros(3, dtype=np.float32)
            if pulse is not None and t < pulse_width:
                u_t[pulse[0]] = pulse[1]
            u_rows.append(u_t)
            y_rows.append(state.copy())

    for bit, sign in enumerate(current):
        append_steps(pulse_width, (bit, float(sign)))
        append_steps(int(rng.integers(2, min_interval + 1)))

    while len(u_rows) < steps:
        directed_edges = [(mem, bit) for mem in memories for bit in range(3)]
        rng.shuffle(directed_edges)
        for mem, bit in directed_edges:
            mem_arr = np.array(mem, dtype=np.float32)
            for route_bit in rng.permutation(3):
                if state[route_bit] != mem_arr[route_bit]:
                    append_steps(
                        pulse_width, (int(route_bit), float(mem_arr[route_bit]))
                    )
                    append_steps(int(rng.integers(min_interval, max_interval + 1)))
            flip_sign = float(-state[bit])
            append_steps(pulse_width, (bit, flip_sign))
            append_steps(int(rng.integers(min_interval, max_interval + 1)))
            if len(u_rows) >= steps:
                break

    return torch.from_numpy(np.stack(u_rows[:steps])), torch.from_numpy(
        np.stack(y_rows[:steps])
    )


def make_training_task(
    cfg: Config, rng: np.random.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    if cfg.train_task == "random":
        return make_pulse_task(
            cfg.train_steps, cfg.pulse_width, cfg.min_interval, cfg.max_interval, rng
        )
    if cfg.train_task == "structured":
        return make_balanced_transition_task(
            cfg.train_steps, cfg.pulse_width, cfg.min_interval, cfg.max_interval, rng
        )
    if cfg.train_task == "mixed":
        structured_steps = int(round(cfg.train_steps * cfg.structured_fraction))
        structured_steps = min(max(structured_steps, 0), cfg.train_steps)
        random_steps = cfg.train_steps - structured_steps
        parts_u: list[torch.Tensor] = []
        parts_y: list[torch.Tensor] = []
        if structured_steps:
            u_s, y_s = make_balanced_transition_task(
                structured_steps,
                cfg.pulse_width,
                cfg.min_interval,
                cfg.max_interval,
                rng,
            )
            parts_u.append(u_s)
            parts_y.append(y_s)
        if random_steps:
            initial_state = parts_y[-1][-1].numpy() if parts_y else None
            u_r, y_r = make_pulse_task(
                random_steps,
                cfg.pulse_width,
                cfg.min_interval,
                cfg.max_interval,
                rng,
                initial_state,
            )
            parts_u.append(u_r)
            parts_y.append(y_r)
        return torch.cat(parts_u, dim=0), torch.cat(parts_y, dim=0)
    raise ValueError("train_task must be one of: random, structured, mixed")


def init_network(cfg: Config) -> dict[str, torch.Tensor]:
    n = cfg.n
    dev = resolve_device(cfg.device)
    j = cfg.g * torch.randn(n, n, device=dev, dtype=NETWORK_DTYPE) / math.sqrt(n)
    b = (
        cfg.input_scale
        * torch.randn(n, 3, device=dev, dtype=NETWORK_DTYPE)
        / math.sqrt(3)
    )
    wfb = (
        cfg.feedback_scale
        * torch.randn(n, 3, device=dev, dtype=NETWORK_DTYPE)
        / math.sqrt(3)
    )
    wout = torch.zeros(3, n, device=dev, dtype=NETWORK_DTYPE)
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
    cfg: Config, net: dict[str, torch.Tensor], u: torch.Tensor, target: torch.Tensor
) -> list[float]:
    dev = resolve_device(cfg.device)
    dtype = net["j"].dtype
    u = u.to(dev, dtype=dtype)
    target = target.to(dev, dtype=dtype)
    n = cfg.n
    x = torch.zeros(n, device=dev, dtype=dtype)
    p = torch.eye(n, device=dev, dtype=dtype) / cfg.rls_alpha
    losses: list[float] = []

    for t in range(cfg.train_steps):
        r = torch.tanh(x)
        z = net["wout"] @ r
        dx = -x + net["j"] @ r + net["wfb"] @ z + net["b"] @ u[t]
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
    dev = resolve_device(cfg.device)
    dtype = net["j"].dtype
    u = u.to(dev, dtype=dtype)
    x = (
        torch.zeros(cfg.n, device=dev, dtype=dtype)
        if x0 is None
        else x0.clone().to(dev, dtype=dtype)
    )
    xs, zs = [], []
    for t in range(u.shape[0]):
        x, _, z = step_rnn(x, u[t], net, cfg.dt)
        xs.append(x.detach().cpu())
        zs.append(z.detach().cpu())
    return torch.stack(xs), torch.stack(zs), u.detach().cpu()


@torch.no_grad()
def settle_to_memory(
    cfg: Config, net: dict[str, torch.Tensor], memory: np.ndarray
) -> torch.Tensor:
    dev = resolve_device(cfg.device)
    dtype = net["j"].dtype
    x = torch.zeros(cfg.n, device=dev, dtype=dtype)
    for bit, sign in enumerate(memory):
        u = torch.zeros(1, cfg.pulse_width, 3, device=dev, dtype=dtype)
        u[0, :, bit] = float(sign)
        for t in range(cfg.pulse_width):
            x, _, _ = step_rnn(x, u[0, t], net, cfg.dt)
        for _ in range(cfg.settle_steps // 3):
            x, _, _ = step_rnn(x, torch.zeros(3, device=dev, dtype=dtype), net, cfg.dt)
    for _ in range(cfg.settle_steps):
        x, _, _ = step_rnn(x, torch.zeros(3, device=dev, dtype=dtype), net, cfg.dt)
    return x.detach().cpu()


@torch.no_grad()
def make_transition_trajectories(
    cfg: Config,
    net: dict[str, torch.Tensor],
    memory_x: dict[tuple[int, int, int], torch.Tensor],
    amplitudes: list[float] | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    dev = resolve_device(cfg.device)
    dtype = net["j"].dtype
    transitions: list[dict[str, object]] = []
    detailed: list[dict[str, object]] = []

    for mem, x0_cpu in memory_x.items():
        mem_arr = np.array(mem, dtype=np.float32)
        for bit in range(3):
            target_sign = -mem_arr[bit]
            u = torch.zeros(
                cfg.pulse_width + cfg.transition_relax_steps, 3, device=dev, dtype=dtype
            )
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
        u = torch.zeros(
            cfg.pulse_width + cfg.transition_relax_steps, 3, device=dev, dtype=dtype
        )
        u[: cfg.pulse_width, 0] = amp
        xs, _, uu = simulate(cfg, net, u.cpu(), x0_cpu)
        detailed.append({"amp": amp, "xs": xs, "u": uu})

    return transitions, detailed


def make_edge_interpolation_initial_states(
    cfg: Config, memory_x: dict[tuple[int, int, int], torch.Tensor]
) -> torch.Tensor:
    """Seed q minimization near one-bit basin boundaries.

    The paper sampled initial conditions from transition trajectories. Adding
    straight interpolants between the corresponding attractors keeps the search
    faithful to the same 24 one-bit transition structure while improving the
    chance of landing on separatrix saddles in finite CPU runs.
    """
    if cfg.edge_interpolation_points <= 0:
        first = next(iter(memory_x.values()))
        return first.new_empty((0, first.numel()))

    states: list[torch.Tensor] = []
    alphas = torch.linspace(0.15, 0.85, cfg.edge_interpolation_points)
    for mem, x0 in memory_x.items():
        for bit in range(3):
            # Use each undirected cube edge once.
            if mem[bit] != -1:
                continue
            target = list(mem)
            target[bit] = 1
            x1 = memory_x[tuple(target)]
            for alpha in alphas:
                states.append((1.0 - alpha) * x0 + alpha * x1)
    return (
        torch.stack(states)
        if states
        else next(iter(memory_x.values())).new_empty((0, cfg.n))
    )


def memory_states_from_stable_fixed_points(
    memory_x: dict[tuple[int, int, int], torch.Tensor],
    fps: torch.Tensor,
    fp_info: list[dict[str, object]],
) -> dict[tuple[int, int, int], torch.Tensor]:
    refined = dict(memory_x)
    for i, info in enumerate(fp_info):
        if info["unstable_count"] != 0:
            continue
        mem = tuple(int(v) for v in info["memory"])
        if mem in refined:
            refined[mem] = fps[i].detach().cpu()
    return refined


@torch.no_grad()
def transition_final_sign(
    cfg: Config,
    net: dict[str, torch.Tensor],
    x0_cpu: torch.Tensor,
    bit: int,
    pulse_sign: float,
    amplitude: float,
    relax_extra: int,
) -> int:
    dev = resolve_device(cfg.device)
    dtype = net["j"].dtype
    u = torch.zeros(
        cfg.pulse_width + cfg.transition_relax_steps + relax_extra,
        3,
        device=dev,
        dtype=dtype,
    )
    u[: cfg.pulse_width, bit] = pulse_sign * amplitude
    _, zs, _ = simulate(cfg, net, u.cpu(), x0_cpu)
    sign = int(torch.sign(zs[-1, bit]).item())
    return sign if sign != 0 else 1


@torch.no_grad()
def make_critical_pulse_initial_states(
    cfg: Config,
    net: dict[str, torch.Tensor],
    memory_x: dict[tuple[int, int, int], torch.Tensor],
) -> tuple[torch.Tensor, list[dict[str, object]]]:
    """Sample trajectories near the input-amplitude threshold for each transition."""
    dev = resolve_device(cfg.device)
    dtype = net["j"].dtype
    states: list[torch.Tensor] = []
    thresholds: list[dict[str, object]] = []

    for mem, x0_cpu in memory_x.items():
        for bit in range(3):
            source_sign = int(mem[bit])
            pulse_sign = float(-source_sign)
            lo, hi = 0.0, 1.0
            sign_lo = transition_final_sign(
                cfg, net, x0_cpu, bit, pulse_sign, lo, cfg.critical_relax_extra
            )
            sign_hi = transition_final_sign(
                cfg, net, x0_cpu, bit, pulse_sign, hi, cfg.critical_relax_extra
            )
            if sign_lo != source_sign or sign_hi != -source_sign:
                continue

            for _ in range(cfg.critical_bisection_steps):
                mid = 0.5 * (lo + hi)
                sign_mid = transition_final_sign(
                    cfg, net, x0_cpu, bit, pulse_sign, mid, cfg.critical_relax_extra
                )
                if sign_mid == -source_sign:
                    hi = mid
                else:
                    lo = mid

            amp = 0.5 * (lo + hi)
            thresholds.append({"from": mem, "bit": bit, "amplitude": float(amp)})
            for delta in (0.0, -0.015, 0.015):
                a = min(1.0, max(0.0, amp + delta))
                u = torch.zeros(
                    cfg.pulse_width
                    + cfg.transition_relax_steps
                    + cfg.critical_relax_extra,
                    3,
                    device=dev,
                    dtype=dtype,
                )
                u[: cfg.pulse_width, bit] = pulse_sign * a
                xs, _, _ = simulate(cfg, net, u.cpu(), x0_cpu)
                states.append(xs[:: cfg.critical_candidate_stride])

    if not states:
        first = next(iter(memory_x.values()))
        return first.new_empty((0, first.numel())), thresholds
    return torch.cat(states, dim=0), thresholds


@torch.no_grad()
def evaluate_flipflop_behavior(
    cfg: Config,
    net: dict[str, torch.Tensor],
    memory_x: dict[tuple[int, int, int], torch.Tensor],
) -> dict[str, object]:
    dev = resolve_device(cfg.device)
    dtype = net["j"].dtype
    zero_u = torch.zeros(cfg.settle_steps, 3, device=dev, dtype=dtype)
    memory_ok = 0
    transition_ok = 0

    for mem, x0_cpu in memory_x.items():
        _, z_hold, _ = simulate(cfg, net, zero_u.cpu(), x0_cpu)
        hold_sign = tuple(int(v) for v in torch.sign(z_hold[-1]).cpu().numpy())
        if hold_sign == mem:
            memory_ok += 1

        for bit in range(3):
            target = list(mem)
            target[bit] *= -1
            target_tuple = tuple(target)
            u = torch.zeros(
                cfg.pulse_width + cfg.transition_relax_steps, 3, device=dev, dtype=dtype
            )
            u[: cfg.pulse_width, bit] = float(target_tuple[bit])
            _, z, _ = simulate(cfg, net, u.cpu(), x0_cpu)
            final_sign = tuple(int(v) for v in torch.sign(z[-1]).cpu().numpy())
            transition_ok += int(final_sign == target_tuple)

    return {
        "memory_hold_ok": memory_ok,
        "memory_hold_total": len(memory_x),
        "one_bit_transition_ok": transition_ok,
        "one_bit_transition_total": len(memory_x) * 3,
    }


def autonomous_velocity(x: torch.Tensor, jeff: torch.Tensor) -> torch.Tensor:
    return -x + torch.tanh(x) @ jeff.T


def q_and_grad_numpy(x: np.ndarray, jeff: np.ndarray) -> tuple[float, np.ndarray]:
    r = np.tanh(x)
    f = -x + jeff @ r
    q = 0.5 * np.mean(f * f)
    grad = (-f + (1.0 - r * r) * (jeff.T @ f)) / x.size
    return float(q), grad.astype(np.float64, copy=False)


def refine_fixed_points_lbfgs(
    cfg: Config, jeff: torch.Tensor, xs: torch.Tensor, q: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if cfg.fixed_point_refine <= 0 or xs.numel() == 0:
        return xs.new_empty((0, xs.shape[1] if xs.ndim == 2 else 0)), q.new_empty((0,))

    jeff_np = jeff.detach().cpu().double().numpy()
    order = torch.argsort(q)[: min(cfg.fixed_point_refine, len(q))].tolist()
    refined_x: list[np.ndarray] = []
    refined_q: list[float] = []

    for idx in order:
        x0 = xs[idx].detach().cpu().double().numpy()

        def objective(v: np.ndarray) -> tuple[float, np.ndarray]:
            return q_and_grad_numpy(v, jeff_np)

        res = minimize(
            objective,
            x0,
            method="L-BFGS-B",
            jac=True,
            options={
                "maxiter": cfg.fixed_point_refine_iter,
                "ftol": 1e-15,
                "gtol": 1e-10,
                "maxls": 40,
            },
        )
        q_val, _ = q_and_grad_numpy(res.x, jeff_np)
        refined_x.append(
            res.x.astype(np.float32 if xs.dtype == torch.float32 else np.float64)
        )
        refined_q.append(q_val)

    return torch.from_numpy(np.stack(refined_x)), torch.tensor(refined_q, dtype=q.dtype)


def find_fixed_points(
    cfg: Config, net: dict[str, torch.Tensor], initial_states: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    dev = resolve_device(cfg.device)
    dtype = net["j"].dtype
    j = net["j"].to(dev, dtype=dtype)
    wfb = net["wfb"].to(dev, dtype=dtype)
    wout = net["wout"].to(dev, dtype=dtype)
    jeff = j + wfb @ wout

    if initial_states.shape[0] > cfg.fixed_point_ics:
        idx = torch.randperm(initial_states.shape[0])[: cfg.fixed_point_ics]
        initial_states = initial_states[idx]
    x = initial_states.to(dev, dtype=dtype).clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([x], lr=cfg.fixed_point_lr)
    best_x = x.detach().clone()
    best_q = torch.full((x.shape[0],), float("inf"), device=dev, dtype=dtype)

    decay_points = {
        int(cfg.fixed_point_steps * 0.45),
        int(cfg.fixed_point_steps * 0.75),
    }
    for step in range(cfg.fixed_point_steps):
        opt.zero_grad(set_to_none=True)
        f = autonomous_velocity(x, jeff)
        q = 0.5 * torch.mean(f * f, dim=1)
        loss = q.mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([x], max_norm=10.0)
        opt.step()
        with torch.no_grad():
            improved = q < best_q
            best_q[improved] = q[improved]
            best_x[improved] = x.detach()[improved]
        if step in decay_points:
            for group in opt.param_groups:
                group["lr"] *= 0.35

    with torch.no_grad():
        f = autonomous_velocity(best_x, jeff)
        best_q = 0.5 * torch.mean(f * f, dim=1)

    cpu_x = best_x.detach().cpu()
    cpu_q = best_q.detach().cpu()
    refined_x, refined_q = refine_fixed_points_lbfgs(cfg, jeff, cpu_x, cpu_q)
    all_x = torch.cat([cpu_x, refined_x.to(cpu_x.dtype)], dim=0)
    all_q = torch.cat([cpu_q, refined_q.to(cpu_q.dtype)], dim=0)
    return all_x, all_q


def cluster_fixed_points(
    xs: torch.Tensor, q: torch.Tensor, cfg: Config
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
        dists = torch.stack(
            [torch.linalg.norm(x - c) / math.sqrt(cfg.n) for c in centers]
        )
        if float(dists.min()) > cfg.cluster_distance:
            centers.append(x)
            center_q.append(q[idx])
    if not centers:
        return xs[order[:0]], q[order[:0]]
    return torch.stack(centers), torch.stack(center_q)


def pca_fit_transform(
    data: torch.Tensor, n_components: int = 3
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = data.mean(dim=0)
    centered = data - mean
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    components = vh[:n_components]
    return centered @ components.T, mean, components


def project(
    x: torch.Tensor, mean: torch.Tensor, components: torch.Tensor
) -> torch.Tensor:
    return (x - mean) @ components.T


def classify_fixed_points(
    cfg: Config, net: dict[str, torch.Tensor], fps: torch.Tensor
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


def save_figure(fig: plt.Figure, out_path: Path, show: bool = False) -> None:
    fig.savefig(out_path, dpi=300)
    if show:
        plt.show()


def trim_spines_to_ticks(ax: plt.Axes) -> None:
    x_ticks = [
        tick for tick in ax.get_xticks() if ax.get_xlim()[0] <= tick <= ax.get_xlim()[1]
    ]
    y_ticks = [
        tick for tick in ax.get_yticks() if ax.get_ylim()[0] <= tick <= ax.get_ylim()[1]
    ]
    if len(x_ticks) >= 2:
        ax.spines["bottom"].set_bounds(x_ticks[0], x_ticks[-1])
    if len(y_ticks) >= 2:
        ax.spines["left"].set_bounds(y_ticks[0], y_ticks[-1])


def add_flow_arrow(
    ax: plt.Axes,
    xy: np.ndarray,
    color,
    frac: float = 0.5,
    size: float = 12.0,
    lw: float = 1.2,
) -> None:
    """Add one tangent arrow at a fractional arc-length position."""
    if len(xy) < 3:
        return
    segment_lengths = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    total_length = float(segment_lengths.sum())
    if total_length < 1e-8:
        return
    target_length = np.clip(frac, 0.0, 1.0) * total_length
    idx = int(np.searchsorted(np.cumsum(segment_lengths), target_length))
    idx = max(0, min(len(xy) - 2, idx))
    p0 = xy[idx]
    p1 = xy[idx + 1]
    if np.linalg.norm(p1 - p0) < 1e-8:
        return
    ax.annotate(
        "",
        xy=(p1[0], p1[1]),
        xytext=(p0[0], p0[1]),
        arrowprops=dict(
            arrowstyle="->",
            color=color,
            lw=lw,
            mutation_scale=size,
            shrinkA=0,
            shrinkB=0,
        ),
        zorder=5,
    )


def plot_task_behavior(
    out_path: Path,
    u: torch.Tensor,
    z: torch.Tensor,
    target: torch.Tensor,
    show: bool = False,
) -> None:
    colors = [("#7f1515", "#ff2020"), ("#0f3b1d", "#19a35b"), ("#173a92", "#315fd6")]
    t = np.arange(u.shape[0])
    fig = plt.figure(figsize=(7.2, 7.2))
    offsets = np.array([3.6, 0.0, -3.6])
    for k, (dark, light) in enumerate(colors):
        input_label = "input" if k == 0 else None
        output_label = "output" if k == 0 else None
        target_label = "target" if k == 0 else None
        plt.plot(
            t,
            u[:, k].numpy() * 0.65 + offsets[k],
            color=dark,
            lw=1.4,
            label=input_label,
        )
        plt.plot(
            t,
            z[:, k].numpy() * 0.75 + offsets[k],
            color=light,
            lw=1.7,
            label=output_label,
        )
        plt.plot(
            t,
            target[:, k].numpy() * 0.75 + offsets[k],
            color=light,
            lw=0.8,
            alpha=0.25,
            label=target_label,
        )
        plt.text(-35, offsets[k] + 0.72, "+1", fontsize=9, ha="right", va="center")
        plt.text(-35, offsets[k] - 0.72, "-1", fontsize=9, ha="right", va="center")
    plt.xlim(0, len(t))

    plt.yticks([])
    plt.xlabel("Time")
    plt.title("3-bit flip-flop task behavior", pad=24)
    plt.legend(loc="upper right", bbox_to_anchor=(1.05, 1.05))
    plt.gca().spines[["top", "right", "left"]].set_visible(False)

    save_figure(fig, out_path, show=show)
    plt.close(fig)


def save_model_behavior_plot_data(
    out_path: Path,
    u: torch.Tensor,
    z: torch.Tensor,
    target: torch.Tensor,
    train_losses: list[float],
) -> None:
    torch.save(
        {
            "u": u.detach().cpu().to(torch.float32),
            "z": z.detach().cpu().to(torch.float32),
            "target": target.detach().cpu().to(torch.float32),
            "train_losses": [float(v) for v in train_losses],
        },
        out_path,
    )


def render_model_behavior_plot_data(
    out_path: Path, data: dict[str, object], show: bool = False
) -> None:
    plot_task_behavior(out_path, data["u"], data["z"], data["target"], show=show)


def render_model_behavior_from_model(
    out_path: Path, cfg_path: Path, model_path: Path, show: bool = False
) -> None:
    cfg = load_config_json(cfg_path)
    cfg.device = "cpu"
    rng = np.random.default_rng(cfg.seed)
    u, target = make_pulse_task(
        cfg.test_steps, cfg.pulse_width, cfg.min_interval, cfg.max_interval, rng
    )
    net = load_model(model_path)
    _, z, u = simulate(cfg, net, u)
    plot_task_behavior(out_path, u, z, target, show=show)


def build_state_space_plot_data(
    cfg: Config,
    net: dict[str, torch.Tensor],
    transitions: list[dict[str, object]],
    detailed: list[dict[str, object]],
    fps: torch.Tensor,
    fp_info: list[dict[str, object]],
) -> dict[str, object]:
    transition_states = torch.cat([item["xs"][::5] for item in transitions], dim=0)
    _, mean, comps = pca_fit_transform(transition_states, 3)
    fp_proj = project(fps, mean, comps).detach().cpu().to(torch.float32)

    left_transition_curves = [
        project(item["xs"], mean, comps).detach().cpu().to(torch.float32)
        for item in transitions
    ]

    stable_idx = [i for i, info in enumerate(fp_info) if info["unstable_count"] == 0]
    saddle_idx = [i for i, info in enumerate(fp_info) if info["unstable_count"] == 1]
    other_idx = [i for i, info in enumerate(fp_info) if info["unstable_count"] > 1]
    left_points = {
        "stable": fp_proj[stable_idx] if stable_idx else torch.empty(0, 3),
        "saddle": fp_proj[saddle_idx] if saddle_idx else torch.empty(0, 3),
        "other": fp_proj[other_idx] if other_idx else torch.empty(0, 3),
    }

    saddle_trace_steps = max(cfg.transition_relax_steps * 4, cfg.settle_steps * 2, 640)
    saddle_perturbation = 0.25
    unstable_lines: list[torch.Tensor] = []
    saddle_traces: list[torch.Tensor] = []
    for i, info in enumerate(fp_info):
        vecs = info["unstable_vectors"]
        if vecs.shape[1] == 0:
            continue
        for k in range(vecs.shape[1]):
            v = torch.from_numpy(vecs[:, k])
            v = v / (torch.linalg.norm(v) + 1e-8)
            line = torch.stack([fps[i] - 1.2 * v, fps[i] + 1.2 * v])
            unstable_lines.append(
                project(line, mean, comps).detach().cpu().to(torch.float32)
            )

            if info["unstable_count"] == 1 and k == 0:
                for sign in [-1.0, 1.0]:
                    x0 = fps[i] + sign * saddle_perturbation * v
                    u0 = torch.zeros(saddle_trace_steps, 3, dtype=net["j"].dtype)
                    xs, _, _ = simulate(cfg, net, u0, x0)
                    saddle_traces.append(
                        project(xs, mean, comps).detach().cpu().to(torch.float32)
                    )

    stable_by_memory = {
        tuple(int(v) for v in fp_info[i]["memory"]): fps[i].detach().cpu()
        for i in stable_idx
    }
    connected_example: dict[str, object] | None = None
    connection_steps = max(cfg.transition_relax_steps * 10, cfg.settle_steps * 4, 1600)
    for i in saddle_idx:
        vecs = fp_info[i]["unstable_vectors"]
        if vecs.shape[1] == 0:
            continue
        v = torch.from_numpy(vecs[:, 0]).to(fps[i].dtype)
        v = v / (torch.linalg.norm(v) + 1e-8)
        endpoint_memories: list[tuple[int, int, int]] = []
        for sign in [-1.0, 1.0]:
            x0 = fps[i] + sign * saddle_perturbation * v
            u0 = torch.zeros(connection_steps, 3, dtype=net["j"].dtype)
            _, zs, _ = simulate(cfg, net, u0, x0)
            endpoint_memories.append(tuple(int(x) for x in torch.sign(zs[-1]).numpy()))
        a, b = endpoint_memories
        if a == b or a not in stable_by_memory or b not in stable_by_memory:
            continue
        diff_bits = [bit for bit in range(3) if a[bit] != b[bit]]
        if len(diff_bits) != 1:
            continue
        bit = diff_bits[0]
        source, target = (a, b) if b[bit] == 1 else (b, a)
        pulse_sign = float(target[bit])
        relax_steps = cfg.pulse_width + max(
            cfg.transition_relax_steps * 3, cfg.settle_steps * 2
        )

        def final_memory_for(
            src: tuple[int, int, int], dst: tuple[int, int, int]
        ) -> tuple[int, int, int]:
            u = torch.zeros(relax_steps, 3, dtype=net["j"].dtype)
            u[: cfg.pulse_width, bit] = float(dst[bit])
            _, zs, _ = simulate(cfg, net, u, stable_by_memory[src])
            return tuple(int(x) for x in torch.sign(zs[-1]).numpy())

        if final_memory_for(source, target) != target:
            source, target = target, source
            pulse_sign = float(target[bit])
            if final_memory_for(source, target) != target:
                continue

        connected_example = {
            "saddle_index": i,
            "saddle": fps[i].detach().cpu(),
            "source": source,
            "target": target,
            "bit": bit,
            "pulse_sign": pulse_sign,
            "relax_steps": relax_steps,
        }
        break

    right_curves: list[dict[str, object]] = []
    right_points: list[dict[str, object]] = []
    if connected_example is not None:
        bit = int(connected_example["bit"])
        source = connected_example["source"]
        target = connected_example["target"]
        pulse_sign = float(connected_example["pulse_sign"])
        relax_steps = int(connected_example["relax_steps"])
        saddle_point = connected_example["saddle"]
        saddle_projected = project(saddle_point[None, :], mean, comps).detach().cpu()[0]
        for amp in np.linspace(0.0, 1.0, 11):
            u = torch.zeros(relax_steps, 3, dtype=net["j"].dtype)
            u[: cfg.pulse_width, bit] = pulse_sign * float(amp)
            xs, _, uu = simulate(cfg, net, u, stable_by_memory[source])
            pts = project(xs, mean, comps).detach().cpu()
            inp = torch.full((pts.shape[0],), float(amp), dtype=pts.dtype)
            coords = torch.stack([pts[:, 2], inp, pts[:, 0]], dim=1).to(torch.float32)
            is_full = abs(float(amp) - 1.0) < 1e-6
            right_curves.append(
                {
                    "coords": coords,
                    "input_amplitude": float(amp),
                    "color": "#2252c7" if is_full else "#52c7dc",
                    "lw": 1.8 if is_full else 0.85,
                    "alpha": 0.9,
                }
            )

        for mem in (source, target):
            p = project(stable_by_memory[mem][None, :], mean, comps).detach().cpu()[0]
            right_points.append(
                {
                    "coords": torch.tensor([p[2], 0.0, p[0]], dtype=torch.float32),
                    "kind": "stable",
                }
            )
        p = saddle_projected
        right_points.append(
            {
                "coords": torch.tensor([p[2], 0.0, p[0]], dtype=torch.float32),
                "kind": "saddle",
            }
        )
        right_ylabel = f"input {bit + 1}"
        right_title = f"Input-amplitude transition {source} -> {target}"
    else:
        for item in detailed:
            pts = project(item["xs"], mean, comps).detach().cpu()
            inp = torch.full((pts.shape[0],), float(item["amp"]), dtype=pts.dtype)
            coords = torch.stack([pts[:, 2], inp, pts[:, 0]], dim=1).to(torch.float32)
            is_full = abs(float(item["amp"]) - 1.0) < 1e-6
            right_curves.append(
                {
                    "coords": coords,
                    "input_amplitude": float(item["amp"]),
                    "color": "#2252c7" if is_full else "#52c7dc",
                    "lw": 1.7 if is_full else 0.9,
                    "alpha": 0.9,
                }
            )
        right_ylabel = "input 1"
        right_title = "Input-amplitude perturbation"

    return {
        "num_fixed_points": int(len(fps)),
        "stable_count": len(stable_idx),
        "saddle_count": len(saddle_idx),
        "other_count": len(other_idx),
        "left": {
            "transition_curves": left_transition_curves,
            "points": left_points,
            "unstable_lines": unstable_lines,
            "saddle_traces": saddle_traces,
        },
        "right": {
            "curves": right_curves,
            "points": right_points,
            "ylabel": right_ylabel,
            "title": right_title,
        },
    }


def draw_fixed_points_transition_panel(
    ax: plt.Axes, data: dict[str, object], elev: float = 18.0, azim: float = -62.0
) -> None:
    left = data["left"]
    for pts_t in left["transition_curves"]:
        pts = pts_t.numpy()
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color="#2f61d5", lw=0.65, alpha=0.8)

    points = left["points"]
    if points["stable"].numel():
        pts = points["stable"].numpy()
        ax.scatter(*pts.T, marker="x", s=80, c="black", lw=2.0, label="stable")
    if points["saddle"].numel():
        pts = points["saddle"].numpy()
        ax.scatter(*pts.T, marker="x", s=70, c="#2cbf31", lw=2.0, label="1D saddle")
    if points["other"].numel():
        pts = points["other"].numpy()
        ax.scatter(*pts.T, marker="x", s=70, c="#d752a8", lw=2.0, label="other")

    for line_t in left["unstable_lines"]:
        pts = line_t.numpy()
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color="red", lw=1.8, alpha=0.9)
    for trace_t in left["saddle_traces"]:
        pts = trace_t.numpy()
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color="red", lw=0.8, alpha=0.72)

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title("Fixed points and 1-bit transitions", pad=24)
    ax.view_init(elev=elev, azim=azim)
    ax.grid(visible=False)
    ax.legend(loc="upper left", fontsize=10)


def render_fixed_points_transition_plot_data(
    out_path: Path,
    data: dict[str, object],
    elev: float = 18.0,
    azim: float = -62.0,
    show: bool = False,
) -> None:
    fig = plt.figure(figsize=(7.2, 6.0))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    draw_fixed_points_transition_panel(ax, data, elev=elev, azim=azim)
    save_figure(fig, out_path, show=show)
    plt.close(fig)


def render_state_space_plot_data(
    out_path: Path, data: dict[str, object], show: bool = False
) -> None:
    fig = plt.figure(figsize=(7.4, 6.0))
    ax = fig.add_axes([0.10, 0.12, 0.72, 0.78])
    right = data["right"]
    cmap = plt.get_cmap("viridis")
    norm = matplotlib.colors.Normalize(vmin=0.0, vmax=1.0)
    for item in right["curves"]:
        pts = item["coords"].numpy()
        amp = float(item.get("input_amplitude", pts[0, 1]))
        color = cmap(norm(amp))
        xy = np.column_stack([pts[:, 0], pts[:, 2]])
        ax.plot(
            xy[:, 0],
            xy[:, 1],
            color=color,
            lw=item["lw"],
            alpha=item["alpha"],
            zorder=2,
        )
        add_flow_arrow(ax, xy, color=color, frac=0.5, size=12.0, lw=1.2)

    non_stable_points = [item for item in right["points"] if item["kind"] != "stable"]
    stable_points = [item for item in right["points"] if item["kind"] == "stable"]
    for item in non_stable_points + stable_points:
        p = item["coords"].numpy()
        color = "black" if item["kind"] == "stable" else "#2cbf31"
        zorder = 10 if item["kind"] == "stable" else 9
        ax.scatter([p[0]], [p[2]], marker="x", s=70, c=color, lw=2.0, zorder=zorder)

    ax.set_title(right["title"], pad=12)
    ax.set_xlabel("PC3")
    ax.set_ylabel("PC1")
    ax.xaxis.set_major_locator(MultipleLocator(1.0))
    ax.yaxis.set_major_locator(MultipleLocator(3.0))

    ax.spines[["top", "right"]].set_visible(False)
    trim_spines_to_ticks(ax)

    cax = fig.add_axes([0.89, 0.18, 0.03, 0.66])
    colorbar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    colorbar.ax.set_title("input", pad=16)
    colorbar.ax.yaxis.set_ticks_position("right")

    save_figure(fig, out_path, show=show)
    plt.close(fig)


def plot_state_space_analysis(
    model_dir: Path,
    figure_dir: Path,
    cfg: Config,
    net: dict[str, torch.Tensor],
    transitions: list[dict[str, object]],
    detailed: list[dict[str, object]],
    fps: torch.Tensor,
    fp_info: list[dict[str, object]],
) -> None:
    data = build_state_space_plot_data(cfg, net, transitions, detailed, fps, fp_info)
    torch.save(data, model_dir / "state_space_plot_data.pt")
    render_fixed_points_transition_plot_data(
        figure_dir / "fixed_points_1d_transition.png", data
    )
    render_state_space_plot_data(figure_dir / "state_space.png", data)


def redraw_state_space_from_plot_data(
    data_path: Path,
    figure_dir: Path,
    fixed_points_elev: float = 18.0,
    fixed_points_azim: float = -62.0,
) -> None:
    data = torch.load(data_path, map_location="cpu", weights_only=False)
    figure_dir.mkdir(parents=True, exist_ok=True)
    render_fixed_points_transition_plot_data(
        figure_dir / "fixed_points_1d_transition.png",
        data,
        elev=fixed_points_elev,
        azim=fixed_points_azim,
    )
    render_state_space_plot_data(figure_dir / "state_space.png", data)
    print(
        json.dumps(
            {
                "redraw_from_plot_data": str(data_path),
                "outputs": {
                    "fixed_points_transition_png": str(
                        figure_dir / "fixed_points_1d_transition.png"
                    ),
                    "state_space_png": str(figure_dir / "state_space.png"),
                },
                "num_fixed_points": int(data["num_fixed_points"]),
                "fixed_point_counts": {
                    "stable": int(data["stable_count"]),
                    "one_unstable": int(data["saddle_count"]),
                    "more_than_one_unstable": int(data["other_count"]),
                },
                "fixed_points_view": [fixed_points_elev, fixed_points_azim],
            },
            indent=2,
        )
    )


def run(cfg: Config) -> None:
    device = resolve_device(cfg.device)
    set_seed(cfg.seed)
    model_dir = Path(cfg.model_dir)
    figure_dir = Path(cfg.figure_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(cfg.seed)
    net = init_network(cfg)
    u_train, y_train = make_training_task(cfg, rng)
    losses = train_force(cfg, net, u_train, y_train)

    u_test, y_test = make_pulse_task(
        cfg.test_steps, cfg.pulse_width, cfg.min_interval, cfg.max_interval, rng
    )
    xs_test, z_test, u_test = simulate(cfg, net, u_test)
    test_mse = float(torch.mean((z_test - y_test) ** 2))

    memories = all_memories()
    memory_x = {
        mem: settle_to_memory(cfg, net, np.array(mem, dtype=np.float32))
        for mem in memories
    }
    transitions, detailed = make_transition_trajectories(cfg, net, memory_x)

    transition_states = torch.cat([item["xs"][::4] for item in transitions], dim=0)
    memory_states = torch.stack(list(memory_x.values()))
    edge_states = make_edge_interpolation_initial_states(cfg, memory_x)
    midpoint_states = []
    for item in transitions:
        midpoint_states.append(item["xs"][cfg.pulse_width + 10])
        midpoint_states.append(item["xs"][cfg.pulse_width + 35])
    fp_initial = torch.cat(
        [
            transition_states,
            memory_states,
            edge_states,
            torch.stack(midpoint_states),
            xs_test[::8],
        ],
        dim=0,
    )
    fp_candidates, q_candidates = find_fixed_points(cfg, net, fp_initial)
    critical_thresholds: list[dict[str, object]] = []
    if cfg.critical_pulse_ics:
        attractor_cfg = Config(**asdict(cfg))
        attractor_cfg.fixed_point_q_thresh = max(
            cfg.fixed_point_q_thresh, cfg.critical_attractor_q_thresh
        )
        prelim_fps, _ = cluster_fixed_points(fp_candidates, q_candidates, attractor_cfg)
        prelim_info = (
            classify_fixed_points(attractor_cfg, net, prelim_fps)
            if len(prelim_fps)
            else []
        )
        memory_x = memory_states_from_stable_fixed_points(
            memory_x, prelim_fps, prelim_info
        )
        critical_states, critical_thresholds = make_critical_pulse_initial_states(
            cfg, net, memory_x
        )
        if len(critical_states):
            critical_candidates, critical_q = find_fixed_points(
                cfg, net, critical_states
            )
            fp_candidates = torch.cat([fp_candidates, critical_candidates], dim=0)
            q_candidates = torch.cat([q_candidates, critical_q], dim=0)
        transitions, detailed = make_transition_trajectories(cfg, net, memory_x)

    fps, fp_q = cluster_fixed_points(fp_candidates, q_candidates, cfg)
    fp_info = classify_fixed_points(cfg, net, fps) if len(fps) else []
    memory_x = memory_states_from_stable_fixed_points(memory_x, fps, fp_info)
    transitions, detailed = make_transition_trajectories(cfg, net, memory_x)
    behavior = evaluate_flipflop_behavior(cfg, net, memory_x)

    save_model_behavior_plot_data(
        model_dir / "model_behavior_plot_data.pt", u_test, z_test, y_test, losses
    )
    plot_task_behavior(figure_dir / "model_behavior.png", u_test, z_test, y_test)
    if len(fps):
        plot_state_space_analysis(
            model_dir, figure_dir, cfg, net, transitions, detailed, fps, fp_info
        )

    stable_count = sum(1 for item in fp_info if item["unstable_count"] == 0)
    saddle_count = sum(1 for item in fp_info if item["unstable_count"] == 1)
    other_count = sum(1 for item in fp_info if item["unstable_count"] > 1)
    fixed_point_counts = {
        "stable": stable_count,
        "one_unstable": saddle_count,
        "more_than_one_unstable": other_count,
    }
    outputs = {
        "cfg_json": str(model_dir / "cfg.json"),
        "model_pt": str(model_dir / "model.pt"),
        "model_behavior_plot_data": str(model_dir / "model_behavior_plot_data.pt"),
        "fixed_point_analysis_pt": (
            str(model_dir / "fixed_point_analysis.pt") if len(fps) else None
        ),
        "state_space_plot_data": (
            str(model_dir / "state_space_plot_data.pt") if len(fps) else None
        ),
        "model_behavior_png": str(figure_dir / "model_behavior.png"),
        "fixed_points_transition_png": (
            str(figure_dir / "fixed_points_1d_transition.png") if len(fps) else None
        ),
        "state_space_png": str(figure_dir / "state_space.png") if len(fps) else None,
    }

    save_config_json(cfg, model_dir / "cfg.json")
    save_model(net, model_dir / "model.pt")
    if len(fps):
        torch.save(
            {
                "fixed_point_candidates": fp_candidates.detach().cpu(),
                "fixed_point_candidate_q": q_candidates.detach().cpu(),
                "critical_pulse_thresholds": critical_thresholds,
                "fixed_points": fps.detach().cpu(),
                "fixed_point_q": fp_q.detach().cpu(),
                "fixed_point_counts": fixed_point_counts,
            },
            model_dir / "fixed_point_analysis.pt",
        )

    print(
        json.dumps(
            {
                "actual_device": str(device),
                "test_mse": test_mse,
                "behavior": {
                    "memory_hold": f"{behavior['memory_hold_ok']}/{behavior['memory_hold_total']}",
                    "one_bit_transition": f"{behavior['one_bit_transition_ok']}/{behavior['one_bit_transition_total']}",
                },
                "num_fixed_points": int(len(fps)),
                "fixed_point_counts": fixed_point_counts,
                "outputs": outputs,
            },
            indent=2,
        )
    )
