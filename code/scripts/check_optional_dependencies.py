"""Pre-flight optional dependency checks for long DOE-Benchmark runs.

Standalone diagnostics can inspect the documented 18-method ``nature`` fallback
when BoTorch is genuinely absent.  The formal publication sweep should pass
``--require-botorch`` so the 20-method headline protocol cannot silently
degrade.  It should also pass ``--require-formal-case-backends`` so cases whose
public response surface depends on optional simulator packages cannot silently
switch backend, and ``--require-runtime-monitoring`` so RSS traces and memory
soft caps are available for server archives.  A half-installed or
version-incompatible optional stack tends to fail only after the operator
launches a multi-hour sweep.  This script makes that state an early, explicit
error.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import importlib.util
import sys
from pathlib import Path
from typing import Iterable, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BO_MODULES = ("torch", "gpytorch", "botorch")
FORMAL_CASE_BACKEND_MODULES = ("neuralfoil", "aerosandbox")
RUNTIME_MONITORING_MODULES = ("psutil",)
BO_POOLS = {"bo", "nature", "sota", "mf"}
REQUIRED_METHODS = {
    "bo": {"botorch_qei", "botorch_saasbo"},
    "nature": {"botorch_qei", "botorch_saasbo"},
    "sota": {"botorch_qei", "botorch_saasbo", "botorch_turbo"},
    "mf": {"mf_cokriging"},
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pool",
        default="nature",
        choices=["default", "extended", "full", "bo", "nature", "sota", "mf"],
        help="Method pool that the next run will launch.",
    )
    parser.add_argument(
        "--require-botorch",
        action="store_true",
        help="Fail if BoTorch is absent even for the degradable nature pool.",
    )
    parser.add_argument(
        "--expected-methods",
        type=int,
        default=None,
        help=(
            "Fail unless the resolved method pool has exactly this many methods. "
            "Use 20 for the formal nature-pool publication run."
        ),
    )
    parser.add_argument(
        "--require-formal-case-backends",
        action="store_true",
        help=(
            "Fail unless optional simulator packages required by the frozen "
            "formal case response surfaces import successfully.  Currently "
            "this checks neuralfoil and aerosandbox for airfoil_stall_2d."
        ),
    )
    parser.add_argument(
        "--require-runtime-monitoring",
        action="store_true",
        help=(
            "Fail unless runtime-monitoring packages needed for formal "
            "server RSS tracing and memory soft caps import successfully. "
            "Currently this checks psutil."
        ),
    )
    return parser.parse_args(argv)


def _installed_modules(names: Iterable[str] = BO_MODULES) -> dict[str, bool]:
    return {name: importlib.util.find_spec(name) is not None for name in names}


def _version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _probe_botorch_imports() -> list[str]:
    """Import the exact symbols used by ``benchmark.bo_methods``."""
    errors: list[str] = []
    try:
        bo_methods = importlib.import_module("benchmark.bo_methods")
    except Exception as exc:
        return [f"cannot import benchmark.bo_methods: {type(exc).__name__}: {exc}"]

    if not getattr(bo_methods, "_BOTORCH_OK", False):
        original = getattr(bo_methods, "_BOTORCH_IMPORT_ERROR", "unknown error")
        errors.append(f"BoTorch core imports failed: {original}")

    for module_name in BO_MODULES:
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            errors.append(f"cannot import {module_name}: {type(exc).__name__}: {exc}")

    return errors


def _probe_formal_case_backend_imports() -> list[str]:
    """Import optional simulator backends that define formal case surfaces."""
    errors: list[str] = []
    installed = _installed_modules(FORMAL_CASE_BACKEND_MODULES)
    versions = ", ".join(
        f"{name}={_version(name) if present else 'missing'}"
        for name, present in installed.items()
    )
    for module_name, present in installed.items():
        if not present:
            errors.append(f"formal case backend {module_name!r} is missing")
            continue
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            errors.append(
                f"cannot import formal case backend {module_name}: "
                f"{type(exc).__name__}: {exc}"
            )
    if errors:
        errors.insert(0, f"formal case backend versions: {versions}")
    return errors


def _probe_runtime_monitoring_imports() -> list[str]:
    """Import packages required for server runtime/resource traceability."""
    errors: list[str] = []
    installed = _installed_modules(RUNTIME_MONITORING_MODULES)
    versions = ", ".join(
        f"{name}={_version(name) if present else 'missing'}"
        for name, present in installed.items()
    )
    for module_name, present in installed.items():
        if not present:
            errors.append(f"runtime monitoring dependency {module_name!r} is missing")
            continue
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            errors.append(
                f"cannot import runtime monitoring dependency {module_name}: "
                f"{type(exc).__name__}: {exc}"
            )
    if errors:
        errors.insert(0, f"runtime monitoring dependency versions: {versions}")
    return errors


def _pool_method_names(pool: str) -> tuple[list[str], str | None]:
    try:
        from benchmark.methods import list_methods
        return list_methods(pool), None
    except Exception as exc:
        return [], f"cannot instantiate pool {pool!r}: {type(exc).__name__}: {exc}"


def check_pool(
    pool: str,
    *,
    require_botorch: bool = False,
    expected_methods: int | None = None,
    require_formal_case_backends: bool = False,
    require_runtime_monitoring: bool = False,
) -> tuple[int, list[str]]:
    """Return ``(exit_code, messages)`` for the requested pool."""
    messages: list[str] = []
    installed = _installed_modules()
    versions = ", ".join(
        f"{name}={_version(name) if present else 'missing'}"
        for name, present in installed.items()
    )
    messages.append(f"optional dependency versions: {versions}")

    if require_formal_case_backends:
        case_backend_errors = _probe_formal_case_backend_imports()
        if case_backend_errors:
            messages.extend(case_backend_errors)
            messages.append(
                "Formal case backend preflight failed. Fix the optional case "
                "solver environment before launching long sweeps; do not rely "
                "on diagnostic analytical fallbacks for headline results."
            )
            return 1, messages
        case_versions = ", ".join(
            f"{name}={_version(name)}" for name in FORMAL_CASE_BACKEND_MODULES
        )
        messages.append(f"formal case backend preflight PASS: {case_versions}")

    if require_runtime_monitoring:
        monitoring_errors = _probe_runtime_monitoring_imports()
        if monitoring_errors:
            messages.extend(monitoring_errors)
            messages.append(
                "Runtime monitoring preflight failed. Install psutil before "
                "launching formal server sweeps so RSS traces and memory caps "
                "are enforced instead of silently missing."
            )
            return 1, messages
        monitoring_versions = ", ".join(
            f"{name}={_version(name)}" for name in RUNTIME_MONITORING_MODULES
        )
        messages.append(f"runtime monitoring preflight PASS: {monitoring_versions}")

    if pool not in BO_POOLS:
        if expected_methods is not None:
            methods, pool_error = _pool_method_names(pool)
            if pool_error:
                messages.append(pool_error)
                return 1, messages
            if len(methods) != expected_methods:
                messages.append(
                    f"pool={pool!r} resolved to {len(methods)} methods, "
                    f"expected {expected_methods}; observed methods={methods}"
                )
                return 1, messages
        messages.append(f"pool={pool!r} does not require BoTorch; PASS")
        return 0, messages

    if not installed["botorch"]:
        if pool == "nature" and not require_botorch:
            messages.append(
                "BoTorch is absent; nature_pool will intentionally degrade "
                "to the 18-method non-BoTorch diagnostic fallback. The "
                "formal headline sweep must pass --require-botorch."
            )
            if expected_methods is not None and expected_methods != 18:
                messages.append(
                    f"pool='nature' resolved to 18 methods without BoTorch, "
                    f"expected {expected_methods}; FAIL"
                )
                return 1, messages
            return 0, messages
        messages.append(f"pool={pool!r} requires BoTorch-compatible methods; FAIL")
        return 1, messages

    import_errors = _probe_botorch_imports()
    if import_errors:
        messages.extend(import_errors)
        messages.append(
            "BoTorch/GPyTorch/Torch are present but incompatible. Fix the "
            "environment before launching long sweeps. Only standalone "
            "diagnostics should inspect the 18-method non-BoTorch fallback."
        )
        return 1, messages

    methods, pool_error = _pool_method_names(pool)
    if pool_error:
        messages.append(pool_error)
        return 1, messages

    names = set(methods)
    required = REQUIRED_METHODS[pool]
    missing = sorted(required - names)
    if missing:
        messages.append(
            f"pool={pool!r} is missing required optional methods {missing}; "
            f"observed methods={methods}"
        )
        return 1, messages
    if expected_methods is not None and len(methods) != expected_methods:
        messages.append(
            f"pool={pool!r} resolved to {len(methods)} methods, "
            f"expected {expected_methods}; observed methods={methods}"
        )
        return 1, messages

    messages.append(f"pool={pool!r} optional dependency preflight PASS; methods={methods}")
    return 0, messages


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    rc, messages = check_pool(
        args.pool,
        require_botorch=bool(args.require_botorch),
        expected_methods=args.expected_methods,
        require_formal_case_backends=bool(args.require_formal_case_backends),
        require_runtime_monitoring=bool(args.require_runtime_monitoring),
    )
    print("=" * 70)
    print("Optional dependency preflight")
    print("=" * 70)
    for msg in messages:
        prefix = "OK" if rc == 0 else "CHECK"
        print(f"  {prefix}  {msg}")
    print("=" * 70)
    print("Result: PASS" if rc == 0 else "Result: FAIL")
    return rc


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
