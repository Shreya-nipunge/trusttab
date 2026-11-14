def test_trusttab_package():
    import trusttab

    assert trusttab is not None
    assert trusttab.__version__ == "0.1.0"


def test_curated_public_api_exports_working_entry_points():
    from trusttab import api

    assert "logistic_regression" in api.SUPPORTED_MODELS
    assert callable(api.load_german_credit)
    assert callable(api.load_heart_disease)
    assert callable(api.inject_symmetric_label_noise)
    assert callable(api.build_model)
    assert callable(api.evaluate_classifier)
    assert callable(api.calibrate_fitted_classifier)
    assert callable(api.run_robustness_benchmark)
    assert callable(api.generate_robustness_report)


def test_public_api_import_does_not_load_optional_models():
    import os
    from pathlib import Path
    import subprocess
    import sys

    project_root = Path(__file__).resolve().parents[1]
    script = """
import sys
from trusttab import api
assert 'xgboost' not in sys.modules
assert 'tabpfn' not in sys.modules
assert callable(api.build_model)
"""
    environment = os.environ.copy()
    source_path = str(project_root / "src")
    current_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        source_path
        if not current_path
        else source_path + os.pathsep + current_path
    )
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=project_root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
