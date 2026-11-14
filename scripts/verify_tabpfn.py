"""Run a bounded, real-data TabPFN fit using TrustTab's configured pipeline.

The script uses 48 German Credit rows for fitting and 16 for prediction. It
loads model parameters from the core experiment configuration and never edits
that configuration. A missing optional dependency is reported as BLOCKED and
does not trigger an installation.
"""

from __future__ import annotations

import argparse
import ctypes
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _physical_memory_bytes() -> int | None:
    """Return total physical memory where the current OS exposes it safely."""
    if sys.platform == "win32":

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.ullTotalPhys)
        return None

    try:
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, OSError, ValueError):
        return None


def _hardware_metadata(torch_module: Any = None) -> dict[str, Any]:
    gpu: dict[str, Any] = {
        "cuda_available": None,
        "devices": [],
        "detection": "not checked",
    }
    if torch_module is not None:
        try:
            cuda_available = bool(torch_module.cuda.is_available())
            devices = []
            if cuda_available:
                devices = [
                    {
                        "name": torch_module.cuda.get_device_name(index),
                        "total_memory_bytes": int(
                            torch_module.cuda.get_device_properties(index).total_memory
                        ),
                    }
                    for index in range(torch_module.cuda.device_count())
                ]
            gpu = {
                "cuda_available": cuda_available,
                "devices": devices,
                "detection": "PyTorch",
            }
        except Exception as exc:  # Hardware discovery must not hide fit status.
            gpu["detection"] = f"PyTorch query failed: {type(exc).__name__}"

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        if gpu["detection"] == "not checked":
            gpu["detection"] = "PyTorch unavailable and nvidia-smi unavailable"
    else:
        gpu["nvidia_smi_devices"] = [
            line.strip() for line in result.stdout.splitlines() if line.strip()
        ]
        gpu["detection"] = "nvidia-smi"

    return {
        "platform": platform.platform(),
        "processor": platform.processor() or None,
        "logical_cpu_count": os.cpu_count(),
        "total_physical_memory_bytes": _physical_memory_bytes(),
        "gpu": gpu,
    }


def _environment_metadata(torch_module: Any = None) -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "numpy": _version("numpy"),
        "pandas": _version("pandas"),
        "scikit_learn": _version("scikit-learn"),
        "trusttab": _version("trusttab"),
        "tabpfn": _version("tabpfn"),
        "torch": _version("torch"),
        "torch_cuda_build": (
            getattr(torch_module.version, "cuda", None)
            if torch_module is not None
            else None
        ),
    }


def verify(seed: int) -> tuple[dict[str, Any], int]:
    """Run bounded fit/predict checks; return a JSON-safe report and exit code."""
    try:
        import torch
    except ImportError:
        torch = None

    report: dict[str, Any] = {
        "status": "BLOCKED",
        "seed": seed,
        "dataset": "German Credit (OpenML data ID 31)",
        "bounded_rows": {"fit": 48, "predict": 16},
        "environment": _environment_metadata(torch),
        "hardware": _hardware_metadata(torch),
    }

    try:
        from trusttab.data import load_german_credit
        from trusttab.experiments.config import load_experiment_config
        from trusttab.experiments.seeding import derive_seed
        from trusttab.models import build_model

        X, y = load_german_credit()
        sampled_X, _, sampled_y, _ = train_test_split(
            X,
            y,
            train_size=64,
            random_state=seed,
            stratify=y,
        )
        X_fit, X_predict, y_fit, _ = train_test_split(
            sampled_X,
            sampled_y,
            test_size=16,
            random_state=derive_seed(seed, "tabpfn_verification_split"),
            stratify=sampled_y,
        )
        report["data"] = {
            "source_rows": len(X),
            "source_features": X.shape[1],
            "fit_rows": len(X_fit),
            "prediction_rows": len(X_predict),
            "fit_class_counts": {
                str(key): int(value)
                for key, value in y_fit.value_counts().sort_index().items()
            },
        }

        configuration = load_experiment_config(ROOT / "configs" / "experiments.yaml")
        model_params = configuration.get("model_params", {}).get("tabpfn", {})
        model = build_model(
            "tabpfn",
            features=X_fit,
            random_state=seed,
            model_params=model_params,
        )
        report["model_parameters"] = model_params
        report["model_constructed"] = type(model.named_steps["classifier"]).__name__

        fit_started = time.perf_counter()
        try:
            model.fit(X_fit, y_fit)
        except ImportError as exc:
            report["fit_seconds"] = time.perf_counter() - fit_started
            report["blocker"] = f"{type(exc).__name__}: {exc}"
            return report, 2
        report["fit_seconds"] = time.perf_counter() - fit_started

        predict_started = time.perf_counter()
        predictions = np.asarray(model.predict(X_predict))
        probabilities = np.asarray(model.predict_proba(X_predict), dtype=float)
        report["prediction_seconds"] = time.perf_counter() - predict_started

        classes = np.asarray(model.classes_)
        valid = (
            predictions.shape == (len(X_predict),)
            and probabilities.shape == (len(X_predict), len(classes))
            and len(classes) == 2
            and set(classes.tolist()) == {0, 1}
            and np.isfinite(probabilities).all()
            and np.all((probabilities >= 0) & (probabilities <= 1))
            and np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
        )
        if not valid:
            report["status"] = "FAILED"
            report["prediction_shape"] = list(predictions.shape)
            report["probability_shape"] = list(probabilities.shape)
            report["classes"] = classes.tolist()
            return report, 1

        report.update(
            status="VERIFIED",
            prediction_shape=list(predictions.shape),
            probability_shape=list(probabilities.shape),
            classes=classes.tolist(),
            probabilities_finite=True,
            probabilities_in_range=True,
            probabilities_sum_to_one=True,
        )
        return report, 0
    except Exception as exc:
        report["status"] = (
            "BLOCKED" if isinstance(exc, (ImportError, OSError)) else "FAILED"
        )
        report["blocker"] = f"{type(exc).__name__}: {exc}"
        return report, 2 if report["status"] == "BLOCKED" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    report, exit_code = verify(args.seed)
    print(json.dumps(report, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
