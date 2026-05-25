"""Render the fixed-points and one-dimensional-transition figure."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from flipflop_demo import make_config, render_fixed_points_transition_plot_data


@dataclass(frozen=True)
class PlotConfig:
    plot_data: str = "outputs/state_space_plot_data.pt"
    out: str = "outputs/fixed_points_1d_transition.png"
    state_space_left_elev: float = 18.0
    state_space_left_azim: float = -62.0


# Edit this object to change input/output paths.
PLOT_CONFIG = PlotConfig()


def main() -> None:
    cfg = PLOT_CONFIG
    data_path = Path(cfg.plot_data)
    if not data_path.exists():
        raise FileNotFoundError(f"{data_path} does not exist. Run run_flipflop_demo.py first.")
    data = torch.load(data_path, map_location="cpu", weights_only=False)
    out_path = Path(cfg.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    view_cfg = make_config(
        state_space_left_elev=cfg.state_space_left_elev,
        state_space_left_azim=cfg.state_space_left_azim,
    )
    render_fixed_points_transition_plot_data(out_path, view_cfg, data)
    print(f"Saved fixed-points figure: {out_path}")


if __name__ == "__main__":
    main()
