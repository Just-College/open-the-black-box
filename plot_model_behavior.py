"""Render the model behavior figure from cached demo outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from flipflop_demo import (
    render_model_behavior_from_model,
    render_model_behavior_plot_data,
)


@dataclass(frozen=True)
class PlotConfig:
    plot_data: str = "model/model_behavior_plot_data.pt"
    cfg: str = "model/cfg.json"
    model: str = "model/model.pt"
    out: str = "figure/model_behavior.png"
    show: bool = True


# Edit this object to change input/output paths.
PLOT_CONFIG = PlotConfig()


def main() -> None:
    cfg = PLOT_CONFIG
    plot_data_path = Path(cfg.plot_data)
    out_path = Path(cfg.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if plot_data_path.exists():
        data = torch.load(plot_data_path, map_location="cpu", weights_only=False)
        render_model_behavior_plot_data(out_path, data, show=cfg.show)
    else:
        render_model_behavior_from_model(
            out_path, Path(cfg.cfg), Path(cfg.model), show=cfg.show
        )
    print(f"Saved model behavior figure: {out_path}")


if __name__ == "__main__":
    main()
