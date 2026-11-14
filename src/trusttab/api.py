"""Supported convenience API for common TrustTab workflows.

Specialized modules remain importable for advanced use. Names listed here are
the stable, curated entry points intended for applications and examples.
"""

from trusttab.data import (
    build_preprocessor,
    load_adult,
    load_german_credit,
    load_heart_disease,
)
from trusttab.evaluation import (
    adaptive_expected_calibration_error,
    calibrate_classifier,
    calibrate_fitted_classifier,
    evaluate_classifier,
    expected_calibration_error,
    log_loss_score,
    maximum_calibration_error,
)
from trusttab.experiments import (
    load_experiment_config,
    run_robustness_benchmark,
    run_stress_test,
)
from trusttab.experiments.config import expand_experiment_conditions
from trusttab.models import SUPPORTED_MODELS, build_model
from trusttab.noise import (
    apply_feature_shifts,
    inject_symmetric_label_noise,
    validate_feature_shifts,
)
from trusttab.reporting import generate_robustness_report
from trusttab.statistics import (
    analyze_friedman_nemenyi,
    calibration_wilcoxon_results,
    compare_degradation_slopes,
    descriptive_summary,
    run_research_analysis,
)
from trusttab.visualization import (
    plot_metric_comparison,
    plot_probability_distribution,
    plot_reliability_diagram,
    plot_roc_curve,
    save_figure,
)

__all__ = [
    "SUPPORTED_MODELS",
    "adaptive_expected_calibration_error",
    "analyze_friedman_nemenyi",
    "apply_feature_shifts",
    "build_model",
    "build_preprocessor",
    "calibrate_classifier",
    "calibrate_fitted_classifier",
    "calibration_wilcoxon_results",
    "compare_degradation_slopes",
    "descriptive_summary",
    "evaluate_classifier",
    "expand_experiment_conditions",
    "expected_calibration_error",
    "generate_robustness_report",
    "inject_symmetric_label_noise",
    "load_adult",
    "load_experiment_config",
    "load_german_credit",
    "load_heart_disease",
    "log_loss_score",
    "maximum_calibration_error",
    "plot_metric_comparison",
    "plot_probability_distribution",
    "plot_reliability_diagram",
    "plot_roc_curve",
    "run_research_analysis",
    "run_robustness_benchmark",
    "run_stress_test",
    "save_figure",
    "validate_feature_shifts",
]
