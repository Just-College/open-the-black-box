"""Render the input-amplitude state-space figure."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from flipflop_demo import render_state_space_plot_data


@dataclass(frozen=True)
class PlotConfig:
    plot_data: str = "model/state_space_plot_data.pt"
    out: str = "figure/state_space.png"


# Edit this object to change input/output paths.
PLOT_CONFIG = PlotConfig()


def main() -> None:
    cfg = PLOT_CONFIG
    data_path = Path(cfg.plot_data)
    if not data_path.exists():
        raise FileNotFoundError(
            f"{data_path} does not exist. Run run_flipflop_demo.py first."
        )
    data = torch.load(data_path, map_location="cpu", weights_only=False)
    out_path = Path(cfg.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    render_state_space_plot_data(out_path, data)
    print(f"Saved state-space figure: {out_path}")


if __name__ == "__main__":
    main()
