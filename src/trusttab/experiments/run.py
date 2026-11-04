"""Command-line entry point for TrustTab experiments."""

import argparse
import json
from pathlib import Path

import yaml

from trusttab.experiments.benchmark import run_robustness_benchmark
from trusttab.experiments.config import load_experiment_config
from trusttab.experiments.feature_shift_runner import run_feature_shift_test
from trusttab.experiments.runner import run_stress_test
from trusttab.experiments.tracking import (
    compare_runs,
    get_run,
    list_runs,
    verify_reproducibility,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a configured TrustTab stress-test experiment."
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to an experiment YAML configuration.",
    )
    parser.add_argument(
        "--runs-dir", type=Path, default=None, help="Tracked run directory."
    )
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--list-runs", action="store_true", help="List recorded runs.")
    actions.add_argument("--inspect-run", metavar="RUN_ID", help="Inspect one run.")
    actions.add_argument(
        "--compare-runs",
        nargs=2,
        metavar=("RUN_ID_A", "RUN_ID_B"),
        help="Compare metadata and metric results from two runs.",
    )
    actions.add_argument(
        "--verify-run",
        metavar="RUN_ID",
        help="Re-execute a run and verify it against the recorded run.",
    )
    parser.add_argument("--dataset", help="Filter listed runs by dataset.")
    parser.add_argument("--model", help="Filter listed runs by model.")
    parser.add_argument(
        "--status",
        choices=("running", "completed", "failed"),
        help="Filter listed runs.",
    )

    args = parser.parse_args()
    runs_dir = args.runs_dir or Path("experiments/runs")

    if args.list_runs:
        print(
            json.dumps(
                list_runs(
                    runs_dir, dataset=args.dataset, model=args.model, status=args.status
                ),
                indent=2,
            )
        )
        return
    if args.inspect_run:
        print(json.dumps(get_run(runs_dir, args.inspect_run), indent=2, default=str))
        return
    if args.compare_runs:
        print(
            json.dumps(
                compare_runs(runs_dir, *args.compare_runs), indent=2, default=str
            )
        )
        return

    verification_reference = None
    if args.verify_run and args.config is None:
        verification_reference = get_run(runs_dir, args.verify_run)
        original_config = verification_reference["config"]["original"]
        config = verification_reference["config"]["normalized"]
    else:
        config_path = args.config or Path("configs/experiments.yaml")
        try:
            with config_path.open("r", encoding="utf-8") as config_file:
                original_config = yaml.safe_load(config_file)
        except FileNotFoundError:
            # Keep the CLI loader replaceable for embedding and tests. For a
            # normal invocation load_experiment_config reports the missing file.
            original_config = None
        config = load_experiment_config(config_path)
        if original_config is None:
            original_config = config

    normalized_config = config

    output = config["output"]
    output_path = Path(output["directory"]) / output["filename"]

    print("Starting TrustTab experiment")
    dataset_names = config["dataset"].get("names") or [config["dataset"].get("name")]
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
    if args.verify_run:
        # Re-execution should create a new tracked result without overwriting
        # the output configured for the original run.
        common["output_path"] = None
    benchmark_mode = config.get("benchmark_mode")
    if benchmark_mode is None:
        benchmark_mode = (
            "feature_shift_legacy"
            if config.get("feature_shifts")
            else "label_noise_legacy"
        )
    tracking_settings = config.get("tracking", {})
    tracking_enabled = bool(tracking_settings.get("enabled", False))
    run_tracking_dir = args.runs_dir or Path(
        tracking_settings.get("directory", "experiments/runs")
    )
    if tracking_enabled or args.verify_run:
        settings = config.get("benchmark_settings", {})
        results = run_robustness_benchmark(
            dataset_names=dataset_names,
            label_noise_levels=settings.get("label_noise_levels", []),
            feature_shifts=settings.get("feature_shifts"),
            **common,
            tracking_dir=run_tracking_dir,
            experiment_name=(
                tracking_settings.get("experiment_name")
                or (f"verification of {args.verify_run}" if args.verify_run else None)
            ),
            original_config=original_config,
            normalized_config=normalized_config,
        )
    elif benchmark_mode == "unified":
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

    if args.verify_run:
        candidate_run_id = results.attrs["run_id"]
        verification = verify_reproducibility(
            run_tracking_dir, args.verify_run, candidate_run_id
        )
        candidate_dir = Path(results.attrs["run_dir"])
        (candidate_dir / "verification.json").write_text(
            json.dumps(verification, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(verification, indent=2))

    print("\nExperiment completed.")
    print(f"Total result rows: {len(results)}")
    if args.verify_run:
        print(f"Verification run: {results.attrs['run_id']}")
        print(f"Tracked artifacts: {results.attrs['run_dir']}")
    else:
        print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
