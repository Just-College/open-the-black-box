# Opening the Black Box: 3-bit Flip-Flop Reproduction

## Overview

This repository is a simple reproduction of the 3-bit flip-flop example from
Sussillo and Barak, "Opening the Black Box" ([`paper.pdf`](./paper.pdf)). The
code trains a continuous-time echo-state RNN with FORCE learning, searches for
fixed points in the trained autonomous dynamics, and renders Figures 2-3.

The practical target of the current version is to recover the main qualitative
Figure 3 structure: eight stable memory attractors and roughly 12-14
one-dimensional saddle points with meaningful transition topology. The exact
26 fixed points reported in the paper are sensitive to network size, random
seed, training quality, numerical precision, and fixed-point search coverage.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Repository Layout](#repository-layout)
- [Configuration](#configuration)
- [Model And Closed-Loop Dynamics](#model-and-closed-loop-dynamics)
- [Main Code Entry Points](#main-code-entry-points)
- [Training Flow](#training-flow)
- [Fixed-Point And Dynamics Analysis](#fixed-point-and-dynamics-analysis)
- [Outputs And Caches](#outputs-and-caches)
- [Recommended Workflow](#recommended-workflow)

## Quick Start

Run the full training, fixed-point analysis, cache generation, and plotting
pipeline:

```bash
python run_flipflop_demo.py
```

For a quick smoke test, edit `run_flipflop_demo.py` and use:

```python
RUN_CONFIG = make_config(preset="test", model_dir="model", figure_dir="figure")
REDRAW_FROM_PLOT_DATA = False
```

To redraw state-space figures from cached plot data without retraining, set
`REDRAW_FROM_PLOT_DATA = True` in `run_flipflop_demo.py`. Then run:

```bash
python run_flipflop_demo.py
```

Individual plot-only entrypoints are also available:

```bash
python plot_model_behavior.py
python plot_fixed_points_transitions.py
python plot_state_space.py
```

Each standalone plot script has a `PlotConfig.show` switch. Set `show=True` to
call `plt.show()` after saving the figure. In headless environments the code
uses a non-interactive backend; for an interactive window, run with a GUI-capable
Matplotlib backend.

## Repository Layout

```bash
.
├── flipflop_demo.py                  # Shared implementation library
├── run_flipflop_demo.py              # Main training/analysis entrypoint
├── plot_model_behavior.py            # Redraw Figure 2-style behavior plot
├── plot_fixed_points_transitions.py  # Redraw Figure 3 fixed-point panel
├── plot_state_space.py               # Redraw input-amplitude state-space panel
├── model/                            # Saved model, analysis, and plot caches
├── figure/                           # Rendered figures
└── paper.pdf                         # Reference paper
```

The repository intentionally keeps configuration in Python objects rather than
command-line arguments. Edit `RUN_CONFIG` in `run_flipflop_demo.py` or the
`PlotConfig` object in each plot script.

## Configuration

Important parameters live in the `Config` dataclass in `flipflop_demo.py`.
The default run is selected in `run_flipflop_demo.py`:

```python
RUN_CONFIG = make_config(preset="full", model_dir="model", figure_dir="figure")
```

Use `preset="test"` for a small smoke test. Use `REDRAW_FROM_PLOT_DATA = True`
when adjusting plot style from existing cached data.

| Parameter | Role |
| --- | --- |
| `n` | Number of recurrent units |
| `g` | Random recurrent gain |
| `dt` | Euler integration step |
| `train_steps` | Number of FORCE/RLS training steps |
| `train_task` | Training task generator: `random`, `structured`, or `mixed` |
| `structured_fraction` | Fraction of mixed training devoted to structured transitions |
| `pulse_width` | Input pulse duration in timesteps |
| `min_interval`, `max_interval` | Random interval range between pulses |
| `rls_alpha`, `rls_every` | RLS update parameters |
| `settle_steps` | Relaxation length used to settle memory states |
| `transition_relax_steps` | Relaxation length after transition pulses |
| `fixed_point_ics` | Maximum number of fixed-point initial conditions |
| `fixed_point_steps` | Adam optimization steps for fixed-point search |
| `fixed_point_q_thresh` | Threshold for accepting fixed points |
| `cluster_distance` | Distance threshold for merging fixed points |
| `unstable_tol` | Eigenvalue threshold for unstable dimensions |
| `device` | `"auto"`, `"cpu"`, or CUDA device string |

GPU execution is useful for training and fixed-point search, but small numerical
differences between CPU and CUDA can affect borderline fixed-point candidates.
For strict comparisons, keep the same seed, device, dtype, and configuration.

## Model And Closed-Loop Dynamics

The model is an echo-state network:

$$
\frac{dx}{dt} = -x + J \tanh(x) + W_{fb} z + B u \\
z = W_{out} \tanh(x)
$$

The implementation uses Euler integration:

$$
x[t + 1] = (1 - dt) x[t] + dt (J \tanh(x[t]) + W_{fb} z[t] + B u[t]) \\
z[t] = W_{out} \tanh(x[t])
$$

Main variables:

| Symbol | Code | Meaning |
| --- | --- | --- |
| `x` | hidden state | Activation variable used for fixed-point analysis and PCA |
| `r = tanh(x)` | rate state | Nonlinear firing-rate state |
| `u` | input | Three pulse inputs, one per flip-flop bit |
| `z` | output | Three-dimensional memory output |
| `J` | `net["j"]` | Fixed random recurrent weights |
| `B` | `net["b"]` | Fixed random input weights |
| `Wfb` | `net["wfb"]` | Fixed feedback weights from output to recurrent state |
| `Wout` | `net["wout"]` | Trained readout weights |

Only `Wout` is trained. The recurrent, input, and feedback weights are random
and fixed after initialization. The effective zero-input autonomous dynamics
after training are:

$$
\frac{dx}{dt} = -x + (J + W_{fb} W_{out}) \tanh(x)
$$

The trained closed-loop dynamics are therefore analyzed through the effective
recurrent matrix:

$$
J_{eff} = J + W_{fb} W_{out}
$$

The code uses `NETWORK_DTYPE = torch.float64` in `flipflop_demo.py`. This is
important for stable fixed-point search and Jacobian-based classification.

## Main Code Entry Points

| File | Function | Role |
| --- | --- | --- |
| `run_flipflop_demo.py` | `main()` | Main entrypoint for full runs or cache redraws |
| `flipflop_demo.py` | `make_config()` | Builds a `Config` from `full` or `test` presets |
| `flipflop_demo.py` | `make_training_task()` | Builds the training input/target sequence |
| `flipflop_demo.py` | `init_network()` | Initializes random echo-state RNN weights |
| `flipflop_demo.py` | `train_force()` | Trains the readout with FORCE/RLS updates |
| `flipflop_demo.py` | `simulate()` | Runs the trained network forward |
| `flipflop_demo.py` | `settle_to_memory()` | Settles the network near a target memory state |
| `flipflop_demo.py` | `make_transition_trajectories()` | Generates all one-bit transition trajectories |
| `flipflop_demo.py` | `find_fixed_points()` | Optimizes candidate fixed points |
| `flipflop_demo.py` | `cluster_fixed_points()` | Merges duplicate fixed-point candidates |
| `flipflop_demo.py` | `classify_fixed_points()` | Classifies stability by Jacobian eigenvalues |
| `flipflop_demo.py` | `build_state_space_plot_data()` | Creates cached projected data for Figure 3 plots |
| `flipflop_demo.py` | `render_state_space_plot_data()` | Renders the input-amplitude state-space panel |
| `flipflop_demo.py` | `run()` | Full training, analysis, cache, and render pipeline |

## Training Flow

The full run in `run(cfg)` follows this sequence:

1. Resolve device with `cfg.device`. The default `"auto"` uses CUDA when
   available and otherwise falls back to CPU.
2. Set the random seed with `set_seed(cfg.seed)`.
3. Initialize the network with `init_network(cfg)`.
4. Generate a 3-bit flip-flop training task with `make_training_task(cfg, rng)`.
5. Train the readout weights using `train_force(cfg, net, u_train, y_train)`.
6. Simulate a test pulse sequence and render the Figure 2-style behavior plot.
7. Settle the network to the eight target memory states.
8. Generate all 24 one-bit transition trajectories.
9. Search, cluster, and classify fixed points.
10. Cache the analysis and render the Figure 3-style plots.

The default task mode is:

```python
train_task = "mixed"
structured_fraction = 0.7
```

This combines structured one-bit transitions with random pulse sequences. The
structured part helps the trained dynamics cover the transition regions needed
for the Figure 3 fixed-point analysis.

## Fixed-Point And Dynamics Analysis

Fixed points are searched in the zero-input autonomous dynamics after training.
Because `z = Wout tanh(x)`, the autonomous velocity is:

$$
F(x) = -x + (J + W_{fb} W_{out}) \tanh(x)
$$

A fixed point satisfies:

$$
F(x*) = 0
$$

The code minimizes:

$$
q(x) = \frac{1}{2} \|F(x)\|^2
$$

The search is initialized from task-relevant states rather than purely random
states:

- samples from one-bit transition trajectories;
- settled memory states;
- interpolations between memory states;
- test trajectory states;
- optional critical-pulse states near input-driven transition thresholds.

Candidate fixed points are optimized with Adam, refined with LBFGS, filtered by
`fixed_point_q_thresh`, and merged by `cluster_fixed_points()`.

Each clustered fixed point is classified with the Jacobian of the autonomous
dynamics:

$$
\frac{dF}{dx} = (J + W_{fb} W_{out}) \mathrm{diag}(1 - \tanh(x)^2) - I
$$

Classification uses the number of eigenvalues whose real part is greater than
`unstable_tol`:

| Unstable eigenvalues | Interpretation |
| --- | --- |
| 0 | Stable memory attractor |
| 1 | One-dimensional saddle |
| >1 | Higher-dimensional unstable fixed point |

The red trajectories in the fixed-point panel are generated by perturbing fixed
points along their unstable eigenvectors and simulating the autonomous dynamics
forward. For a one-dimensional saddle, the two perturbation signs should ideally
flow toward two different attractors.

## Outputs And Caches

Full runs write to `model/` and `figure/`.

| Path | Meaning |
| --- | --- |
| `model/cfg.json` | Configuration used by the last full run |
| `model/model.pt` | Trained network weights |
| `model/fixed_point_analysis.pt` | Fixed-point candidates, clustered points, and classifications |
| `model/model_behavior_plot_data.pt` | Cached data for the behavior plot |
| `model/state_space_plot_data.pt` | Cached projected data for state-space plots |
| `figure/model_behavior.png` | Figure 2-style input/output behavior |
| `figure/fixed_points_1d_transition.png` | Figure 3 fixed points and transitions |
| `figure/state_space.png` | Input-amplitude transition panel |

For plot style changes, prefer editing render functions or plot scripts and
redrawing from `model/*_plot_data.pt`. This avoids retraining and makes visual
iteration much faster.

## Recommended Workflow

For figure styling:

1. Keep `REDRAW_FROM_PLOT_DATA = True`.
2. Edit `render_state_space_plot_data()`,
   `render_fixed_points_transition_plot_data()`, or `plot_task_behavior()`.
3. Rerun `python run_flipflop_demo.py` or the relevant `plot_*.py` script.

For dynamics or training changes:

1. Set `REDRAW_FROM_PLOT_DATA = False`.
2. Edit `Config`, `make_training_task()`, or training/fixed-point parameters.
3. Run `python run_flipflop_demo.py`.
4. Check the printed fixed-point counts and inspect the figures.
