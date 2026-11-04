
"""Command-line entry point for TrustTab experiments."""

import argparse
from pathlib import Path

from trusttab.experiments.benchmark import run_robustness_benchmark
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
    dataset_names = config["dataset"].get("names") or [
        config["dataset"].get("name")
    ]
    print(f"Dataset(s): {dataset_names}")
    print(f"Models: {config['models']}")
    print(f"Seeds: {config['experiments']['seeds']}")
    print(f"Output: {output_path}")

    common = {
        "seeds": config["experiments"]["seeds"],
        "model_names": config["models"],
        "test_size": config["experiments"].get("test_size", 0.2),
        "output_path": output_path,
    }
    benchmark_mode = config.get("benchmark_mode")
    if benchmark_mode is None:
        benchmark_mode = (
            "feature_shift_legacy"
            if config.get("feature_shifts")
            else "label_noise_legacy"
        )
    if benchmark_mode == "unified":
        settings = config["benchmark_settings"]
        results = run_robustness_benchmark(
            dataset_names=dataset_names,
            label_noise_levels=settings["label_noise_levels"],
            feature_shifts=settings["feature_shifts"],
            **common,
        )
    elif benchmark_mode == "feature_shift_legacy":
        results = run_feature_shift_test(
            dataset_name=dataset_names[0],
            feature_shifts=config["feature_shifts"],
            **common,
        )
    else:
        results = run_stress_test(
            dataset_name=dataset_names[0],
            noise_levels=config["experiments"]["noise_levels"],
            **common,
        )

    print("\nExperiment completed.")
    print(f"Total result rows: {len(results)}")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
