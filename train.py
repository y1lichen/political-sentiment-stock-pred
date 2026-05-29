from pathlib import Path

from src.trump_event_sequence import Config, run_event_sequence


if __name__ == "__main__":
    base_dir = Path.cwd()
    output_dir = base_dir / "output"

    config = Config(
        window_size=20,
        max_splits=7,
        epochs=18,
        batch_size=512,
        hidden_dim=96,
        dropout=0.25,
        high_conf_quantile=0.85,
    )

    run_event_sequence(base_dir=base_dir, output_dir=output_dir, config=config)
