"""Course-friendly entrypoint for the 3-bit flip-flop RNN demo."""

from flipflop_demo import make_config, run


# Edit this object to choose the run mode and parameters. Keeping it here makes
# this file the only training/analysis entrypoint.
RUN_CONFIG = make_config(
    preset="full",
    out_dir="outputs",
    redraw_from_plot_data="outputs/state_space_plot_data.pt",
    state_space_left_elev=18.0,
    state_space_left_azim=-62.0,
)

# For a quick end-to-end check, change the first argument above to
# ``preset="dry_run"`` and clear ``redraw_from_plot_data``.


def main() -> None:
    run(RUN_CONFIG)


if __name__ == "__main__":
    main()
