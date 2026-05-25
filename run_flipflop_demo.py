"""Training entrypoint for the 3-bit flip-flop RNN demo."""

from pathlib import Path

from flipflop_demo import make_config, redraw_state_space_from_plot_data, run

# Edit this object to choose the run mode and parameters. Keeping it here makes
# this file the only training/analysis entrypoint.
RUN_CONFIG = make_config(preset="full", model_dir="model", figure_dir="figure")

# For a quick end-to-end check, change the first argument above to
# ``preset="test"`` and set ``REDRAW_FROM_PLOT_DATA`` to False.
REDRAW_FROM_PLOT_DATA = True
PLOT_DATA = "model/state_space_plot_data.pt"
FIXED_POINTS_ELEV = 18.0
FIXED_POINTS_AZIM = -62.0


def main() -> None:
    if REDRAW_FROM_PLOT_DATA:
        redraw_state_space_from_plot_data(
            Path(PLOT_DATA),
            Path(RUN_CONFIG.figure_dir),
            fixed_points_elev=FIXED_POINTS_ELEV,
            fixed_points_azim=FIXED_POINTS_AZIM,
        )
        return
    run(RUN_CONFIG)


if __name__ == "__main__":
    main()
