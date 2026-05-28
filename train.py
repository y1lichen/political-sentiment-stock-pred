from pathlib import Path

from src.deep_trump_code import Config, run_deep_trump_code


if __name__ == "__main__":
    base_dir = Path.cwd()
    output_dir = base_dir / "output"

    config = Config(
        window_size=20,
        horizons=(1, 2, 3),
        max_splits=8,
        epochs=12,
        batch_size=512,
        hidden_dim=64,
        dropout=0.30,
        min_val_rule_trades=8,
        min_val_rule_hit_rate=0.52,
    )

    run_deep_trump_code(base_dir=base_dir, output_dir=output_dir, config=config)
