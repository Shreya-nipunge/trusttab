
"""Command-line entry point for TrustTab experiments."""

import argparse
from pathlib import Path

from trusttab.experiments.config import load_experiment_config
from trusttab.experiments.runner import run_stress_test


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a configured TrustTab label-noise experiment."
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
    print(f"Noise levels: {config['experiments']['noise_levels']}")
    print(f"Seeds: {config['experiments']['seeds']}")
    print(f"Output: {output_path}")

    results = run_stress_test(
    dataset_name=config["dataset"]["name"],
    noise_levels=config["experiments"]["noise_levels"],
    seeds=config["experiments"]["seeds"],
    model_names=config["models"],
    test_size=config["experiments"]["test_size"],
    output_path=output_path,
)

    print("\nExperiment completed.")
    print(f"Total result rows: {len(results)}")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
