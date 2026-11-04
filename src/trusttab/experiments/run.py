
"""Command-line entry point for TrustTab experiments."""

import argparse
from pathlib import Path

from trusttab.experiments.config import load_experiment_config
from trusttab.experiments.feature_shift_runner import run_feature_shift_test
from trusttab.experiments.runner import run_stress_test


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a configured TrustTab stress-test experiment."
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments.yaml"),
        help="Path to the experiment YAML configuration.",
    )

    args = parser.parse_args()

    config = load_experiment_config(args.config)

    output = config["output"]
    output_path = Path(output["directory"]) / output["filename"]

    print("Starting TrustTab experiment")
    print(f"Dataset: {config['dataset']['name']}")
    print(f"Models: {config['models']}")
    print(f"Seeds: {config['experiments']['seeds']}")
    print(f"Output: {output_path}")

    common = {
        "dataset_name": config["dataset"]["name"],
        "seeds": config["experiments"]["seeds"],
        "model_names": config["models"],
        "test_size": config["experiments"].get("test_size", 0.2),
        "output_path": output_path,
    }
    if config.get("feature_shifts"):
        print(f"Feature shifts: {config['feature_shifts']}")
        results = run_feature_shift_test(
            feature_shifts=config["feature_shifts"],
            **common,
        )
    else:
        print(f"Noise levels: {config['experiments']['noise_levels']}")
        results = run_stress_test(
            noise_levels=config["experiments"]["noise_levels"],
            **common,
        )

    print("\nExperiment completed.")
    print(f"Total result rows: {len(results)}")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
