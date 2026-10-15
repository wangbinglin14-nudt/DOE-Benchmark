"""
benchmark.core
==============
DOE benchmark 套件的核心抽象。

三层架构：
  Sampler    （采样器）   ：在 [0,1]^d 中生成设计矩阵
  Surrogate  （代理模型） ：拟合 (X_train, Y_train)，在 X_test 上预测 (均值, 标准差)
  BenchmarkMethod（方法） ：端到端策略，接收 (case, n_train, X_test, seed)
                            返回 X_test 上的预测结果。

:class:`StaticDesignMethod` 将采样器和代理模型组合，形成标准的
"一次性设计 + 拟合代理模型"流程。Sequential / 主动学习方法直接继承
:class:`BenchmarkMethod`，实现自适应采样循环。
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import socket
import sys
import tempfile
import threading
import time
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

import doe_testbed as db
from benchmark.metrics import compute_all_metrics
from benchmark.runtime_contracts import (
    ALLOWED_FAILURE_CATEGORIES,
    PROTOCOL_DIGEST_COLUMNS,
    RESULT_SCHEMA_VERSION,
    classify_failure_category,
    csv_header_contract_errors,
    failure_category_is_consistent,
    invalid_protocol_digest_mask,
    invalid_runtime_duration_mask,
    protocol_digest_for_row,
    strict_integer_series,
)


# _iter_with_watchdog 用来检测生成器是否耗尽的哨兵值。
# 不直接让 StopIteration 泄漏到 executor future 之外
# （Python 3.7+ 会将这种情况转换为 RuntimeError）。
_END_OF_ITER = object()


_RESUME_REQUIRED_COLUMNS = {
    "case_id",
    "case_source_path",
    "case_source_sha256",
    "domain",
    "dim",
    "n_output",
    "method",
    "method_family",
    "method_source_path",
    "method_source_sha256",
    "budget_mult",
    "n_train",
    "n_test",
    "repeat",
    "seed",
    "test_seed",
    "seed_strategy",
    "seed_start",
    "result_schema_version",
    "protocol_digest",
    "status",
    "failure_category",
    "t_total_s",
    "error",
    "nrmse",
    "worker_host",
    "worker_pid",
    "cell_started_utc",
    "cell_finished_utc",
    "rss_gb_start",
    "rss_gb_end",
    "rss_gb_peak",
}
_RESUME_ALLOWED_STATUSES = {"ok", "timeout", "failed"}
_RESUME_OPTIONAL_OK_NUMERIC_METRICS = ("rmse", "mae", "maxae", "q95ae", "r2")
_REPO_ROOT = Path(__file__).resolve().parents[1]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _process_rss_gb() -> float | None:
    try:
        import psutil  # type: ignore

        return float(psutil.Process().memory_info().rss / (1024 ** 3))
    except Exception:
        return None


def _package_version_status(package: str) -> dict[str, str]:
    """Return a JSON-safe installed/missing package-version record."""
    try:
        return {
            "status": "installed",
            "version": importlib.metadata.version(package),
        }
    except importlib.metadata.PackageNotFoundError:
        return {
            "status": "missing",
            "version": "",
        }
    except Exception as exc:
        return {
            "status": "error",
            "version": "",
            "error": f"{type(exc).__name__}: {exc}",
        }


def _torch_reproducibility_snapshot(*, apply_policy: bool = False) -> dict[str, Any]:
    """Return and optionally enforce the Torch reproducibility policy.

    The benchmark's formal protocol is seed-paired.  BoTorch/GPyTorch methods
    also depend on PyTorch backend switches whose defaults can vary by host,
    CUDA build, or GPU generation.  Capturing these switches in run_config and
    RUN_MANIFEST makes the server archive auditable instead of relying on
    prose claims about deterministic seeds.
    """
    info: dict[str, Any] = {
        "schema_version": 1,
        "policy": {
            "deterministic_algorithms": True,
            "deterministic_warn_only": True,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "allow_tf32": False,
            "float32_matmul_precision": "highest",
            "cublas_workspace_config": ":4096:8",
        },
        "applied": False,
        "status": "not_checked",
        "errors": [],
    }
    try:
        import torch  # type: ignore
    except Exception as exc:
        info["status"] = "missing"
        info["errors"].append(f"{type(exc).__name__}: {exc}")
        return info

    errors: list[str] = []
    if apply_policy:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            try:
                torch.use_deterministic_algorithms(True)
            except Exception as exc:
                errors.append(
                    f"use_deterministic_algorithms failed: {type(exc).__name__}: {exc}"
                )
        except Exception as exc:
            errors.append(
                f"use_deterministic_algorithms failed: {type(exc).__name__}: {exc}"
            )
        try:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        except Exception as exc:
            errors.append(f"cudnn policy failed: {type(exc).__name__}: {exc}")
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
        except Exception as exc:
            errors.append(f"cuda matmul TF32 policy failed: {type(exc).__name__}: {exc}")
        try:
            torch.backends.cudnn.allow_tf32 = False
        except Exception as exc:
            errors.append(f"cudnn TF32 policy failed: {type(exc).__name__}: {exc}")
        setter = getattr(torch, "set_float32_matmul_precision", None)
        if callable(setter):
            try:
                setter("highest")
            except Exception as exc:
                errors.append(
                    f"float32 matmul precision policy failed: {type(exc).__name__}: {exc}"
                )
        info["applied"] = not errors

    def _call_bool(name: str) -> bool | None:
        fn = getattr(torch, name, None)
        if not callable(fn):
            return None
        try:
            return bool(fn())
        except Exception:
            return None

    state: dict[str, Any] = {
        "torch_version": getattr(torch, "__version__", None),
        "cuda_available": None,
        "deterministic_algorithms": _call_bool("are_deterministic_algorithms_enabled"),
        "deterministic_warn_only": _call_bool(
            "is_deterministic_algorithms_warn_only_enabled"
        ),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }
    try:
        state["cuda_available"] = bool(torch.cuda.is_available())
    except Exception:
        state["cuda_available"] = None
    try:
        state["cudnn_benchmark"] = bool(torch.backends.cudnn.benchmark)
        state["cudnn_deterministic"] = bool(torch.backends.cudnn.deterministic)
        state["cudnn_allow_tf32"] = bool(torch.backends.cudnn.allow_tf32)
    except Exception:
        state["cudnn_benchmark"] = None
        state["cudnn_deterministic"] = None
        state["cudnn_allow_tf32"] = None
    try:
        state["cuda_matmul_allow_tf32"] = bool(torch.backends.cuda.matmul.allow_tf32)
    except Exception:
        state["cuda_matmul_allow_tf32"] = None
    getter = getattr(torch, "get_float32_matmul_precision", None)
    if callable(getter):
        try:
            state["float32_matmul_precision"] = str(getter())
        except Exception:
            state["float32_matmul_precision"] = None
    else:
        state["float32_matmul_precision"] = None
    info["state"] = state
    info["errors"].extend(errors)
    info["status"] = "ok" if not errors else "error"
    return info


def _runtime_environment_snapshot() -> dict[str, Any]:
    """Return the Stage-2 startup environment needed to interpret partial runs."""
    return {
        "schema_version": 1,
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "node": socket.gethostname(),
        "cwd": str(Path.cwd()),
        "packages": {
            package: _package_version_status(package)
            for package in (
                "numpy",
                "pandas",
                "scikit-learn",
                "scipy",
                "torch",
                "gpytorch",
                "botorch",
                "neuralfoil",
                "aerosandbox",
                "psutil",
            )
        },
        "torch_reproducibility": _torch_reproducibility_snapshot(
            apply_policy=False
        ),
    }


_TEMP_STORAGE_ENV_VARS = (
    "TMPDIR",
    "TMP",
    "TEMP",
    "TEMPDIR",
    "SLURM_TMPDIR",
    "SCRATCH",
    "LOCAL_SCRATCH",
    "JOBSCRATCH",
)


def _temp_storage_snapshot() -> dict[str, Any]:
    """Return temp/scratch storage context captured at Stage-2 startup."""
    resolved = Path(tempfile.gettempdir()).expanduser()
    info: dict[str, Any] = {
        "schema_version": 1,
        "resolved_path": str(resolved),
        "exists": resolved.exists(),
        "writable": os.access(resolved, os.W_OK) if resolved.exists() else False,
        "free_gb": None,
        "env": {name: os.environ.get(name) for name in _TEMP_STORAGE_ENV_VARS},
    }
    try:
        if resolved.exists():
            usage = shutil.disk_usage(resolved)
            info["free_gb"] = round(float(usage.free / (1024 ** 3)), 6)
    except Exception as exc:
        info["free_gb_error"] = f"{type(exc).__name__}: {exc}"
    return info


def _merge_method_info(row: dict, info: object) -> None:
    """Merge method-provided diagnostics without overriding runner provenance."""
    if not isinstance(info, dict):
        return
    for key, value in info.items():
        if key in row or key in PROTOCOL_DIGEST_COLUMNS:
            continue
        row[key] = value


def _flush_and_fsync(fh) -> None:
    """Persist a just-written stream so crash recovery sees complete rows."""
    fh.flush()
    os.fsync(fh.fileno())


def _fsync_parent_dir(path: Path) -> None:
    """Best-effort directory fsync after atomic replace on platforms that allow it."""
    try:
        dir_fd = os.open(str(path.parent), os.O_RDONLY)
    except Exception:
        return
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON via same-directory temp file + atomic replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        newline="\n",
        dir=str(path.parent),
        prefix=f".{path.name}.tmp.",
        suffix=".json",
        delete=False,
    ) as tmp:
        tmp_path = Path(tmp.name)
        json.dump(payload, tmp, indent=2, sort_keys=True, ensure_ascii=True)
        tmp.write("\n")
        _flush_and_fsync(tmp)
    try:
        os.replace(tmp_path, path)
        _fsync_parent_dir(path)
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink()


class _RssPeakSampler:
    """Sample process RSS during a benchmark cell and report the peak."""

    def __init__(self, interval_s: float = 0.25):
        self.interval_s = float(interval_s)
        self.peak_gb = _process_rss_gb()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self) -> "_RssPeakSampler":
        self._thread.start()
        return self

    def __exit__(self, *exc_info) -> None:
        self._stop.set()
        self._thread.join(timeout=max(self.interval_s * 2.0, 0.1))
        self.observe()

    def observe(self) -> float | None:
        rss = _process_rss_gb()
        if rss is not None:
            self.peak_gb = rss if self.peak_gb is None else max(self.peak_gb, rss)
        return self.peak_gb

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            self.observe()


def case_source_fingerprint(case) -> dict[str, str]:
    """Return source path and SHA-256 for a loaded simulation case."""
    module = __import__(type(case).__module__, fromlist=["__file__"])
    source = Path(str(module.__file__)).resolve()
    h = hashlib.sha256()
    with source.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    try:
        rel = source.relative_to(Path(__file__).resolve().parents[1]).as_posix()
    except ValueError:
        rel = source.as_posix()
    return {"case_source_path": rel, "case_source_sha256": h.hexdigest()}


def _source_fingerprint(obj: object) -> tuple[str, str]:
    module = __import__(type(obj).__module__, fromlist=["__file__"])
    source = Path(str(module.__file__)).resolve()
    h = hashlib.sha256()
    with source.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    try:
        rel = source.relative_to(Path(__file__).resolve().parents[1]).as_posix()
    except ValueError:
        rel = source.as_posix()
    return rel, h.hexdigest()


def _public_config(obj: object) -> dict[str, Any]:
    config: dict[str, Any] = {}

    def _json_canonical(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(k): _json_canonical(v) for k, v in sorted(value.items())}
        if isinstance(value, (tuple, list)):
            return [_json_canonical(v) for v in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    getter = getattr(obj, "get_init_kwargs", None)
    if callable(getter):
        try:
            raw = getter()
            if isinstance(raw, dict):
                config.update({str(key): _json_canonical(value) for key, value in raw.items()})
        except Exception:
            pass
    for key, value in getattr(obj, "__dict__", {}).items():
        if key.startswith("_"):
            continue
        if key in config:
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            config[key] = value
        elif isinstance(value, (tuple, list)) and all(
            isinstance(v, (str, int, float, bool)) or v is None for v in value
        ):
            config[key] = list(value)
    return {str(key): _json_canonical(value) for key, value in config.items()}


def method_source_components(method) -> list[dict[str, Any]]:
    """Return source/config components that define a benchmark method.

    Static-design methods are implemented jointly by the method wrapper,
    sampler, and surrogate classes.  The component list is intentionally
    JSON-serializable so it can be stored in ``RUN_MANIFEST.json`` for
    reviewer-readable method configuration provenance.
    """
    components: list[dict[str, Any]] = []
    for obj in [method, getattr(method, "sampler", None), getattr(method, "surrogate", None)]:
        if obj is None:
            continue
        path, sha = _source_fingerprint(obj)
        item = {
            "class": f"{type(obj).__module__}.{type(obj).__name__}",
            "path": path,
            "sha256": sha,
            "config": _public_config(obj),
        }
        if item not in components:
            components.append(item)
    components.sort(key=lambda x: (str(x["path"]), str(x["class"])))
    return components


def method_source_fingerprint(method) -> dict[str, str]:
    """Return a composite source fingerprint for a benchmark method.

    Static-design methods are implemented jointly by the method wrapper,
    sampler, and surrogate classes, so their digest is a deterministic
    combination of all component source hashes and public configurations.
    """
    components = method_source_components(method)
    digest = hashlib.sha256()
    payload = json.dumps(components, sort_keys=True, separators=(",", ":"), default=str)
    digest.update(payload.encode("utf-8"))
    return {
        "method_source_path": ";".join(str(item["path"]) for item in components),
        "method_source_sha256": digest.hexdigest(),
    }


def _coerce_int(value: object) -> int | None:
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    if isinstance(value, (bool, np.bool_)):
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        if np.isfinite(value) and float(value).is_integer():
            return int(value)
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = float(text)
    except ValueError:
        return None
    if not np.isfinite(parsed) or not parsed.is_integer():
        return None
    return int(parsed)


def _same_resume_value(observed: object, expected: object) -> bool:
    try:
        if pd.isna(observed):
            return False
    except TypeError:
        pass
    if isinstance(expected, (int, np.integer)):
        return _coerce_int(observed) == int(expected)
    return str(observed) == str(expected)


def _normalize_resume_source_path(value: object) -> str:
    text = str(value).strip().replace("\\", "/")
    if not text:
        return ""
    try:
        path = Path(text)
        if path.is_absolute():
            try:
                return path.resolve().relative_to(_REPO_ROOT).as_posix()
            except (OSError, ValueError):
                return path.as_posix()
    except (OSError, ValueError):
        pass
    return text


def _same_resume_source_path(observed: object, expected: object) -> bool:
    observed_text = _normalize_resume_source_path(observed)
    expected_text = _normalize_resume_source_path(expected)
    return bool(observed_text and expected_text) and (
        observed_text == expected_text or observed_text.endswith(f"/{expected_text}")
    )


def _same_resume_contract_value(col: str, observed: object, expected: object) -> bool:
    if col in {"case_source_path", "method_source_path"}:
        return _same_resume_source_path(observed, expected)
    return _same_resume_value(observed, expected)


def _normalize_resume_output_dir_path(value: object) -> str:
    text = str(value).strip()
    if not text:
        return ""
    try:
        return Path(text).expanduser().resolve().as_posix()
    except Exception:
        return Path(text).expanduser().as_posix()


def _resume_int_set(series: pd.Series, field: str, problems: list[str]) -> set[int]:
    values: set[int] = set()
    invalid: list[object] = []
    for raw in series.dropna():
        parsed = _coerce_int(raw)
        if parsed is None:
            invalid.append(raw)
        else:
            values.add(parsed)
    if invalid:
        preview = [str(item) for item in invalid[:10]]
        problems.append(f"non-integer {field} values={preview}")
    return values


def _call_with_soft_timeout(
    fn: Callable[..., Any],
    *args,
    timeout_s: Optional[float],
    **kwargs,
) -> Tuple[Any, Optional[str]]:
    """带可选软超时的函数调用。

    成功时返回 ``(result, None)``，超时/失败时返回 ``(None, reason)``，
    调用者无需 try/except 即可分支处理。

    所谓"软"超时，是指底层线程无法从外部被杀死——Python 缺少可移植的
    线程取消原语——因此函数可能在调用返回后继续在后台运行。仅用于
    后续运行无害的可调用对象（runner 会丢弃它们的结果）。
    """
    if not timeout_s or timeout_s <= 0:
        try:
            return fn(*args, **kwargs), None
        except Exception as exc:
            return None, f"{type(exc).__name__}: {exc}"

    from concurrent.futures import (
        ThreadPoolExecutor,
        TimeoutError as _FutTimeout,
    )
    ex = ThreadPoolExecutor(max_workers=1)
    try:
        fut = ex.submit(fn, *args, **kwargs)
        try:
            return fut.result(timeout=float(timeout_s)), None
        except _FutTimeout:
            return None, f"timeout after {timeout_s:.0f}s"
        except Exception as exc:
            return None, f"{type(exc).__name__}: {exc}"
    finally:
        ex.shutdown(wait=False)


def _iter_with_watchdog(
    gen: Iterable[Any],
    watchdog_s: Optional[float],
) -> Iterator[Any]:
    """从 ``gen`` 中 yield 元素，但如果长时间没有新元素则退出。

    防御 ``joblib.Parallel(return_as='generator')`` 静默卡死的场景——
    例如 loky worker 在损坏的 MKL handle 上进入内核级等待，既不返回
    也不抛出 :class:`TerminatedWorkerError`。没有这个 watchdog，
    ``for row in gen`` 会永远阻塞。

    卡死时抛出 :class:`RuntimeError`，调用者可以回退到串行路径。
    ``watchdog_s = None`` 禁用检查，等价于 ``yield from gen``。
    """
    if not watchdog_s or watchdog_s <= 0:
        yield from gen
        return

    from concurrent.futures import (
        ThreadPoolExecutor,
        TimeoutError as _FutTimeout,
    )
    iterator = iter(gen)
    ex = ThreadPoolExecutor(max_workers=1)
    try:
        while True:
            fut = ex.submit(next, iterator, _END_OF_ITER)
            try:
                row = fut.result(timeout=float(watchdog_s))
            except _FutTimeout:
                raise RuntimeError(
                    f"Parallel generator stalled for {watchdog_s:.0f}s "
                    "(no row received); aborting partition."
                )
            if row is _END_OF_ITER:
                return
            yield row
    finally:
        ex.shutdown(wait=False)


class _Heartbeat:
    """后台守护线程，定期打印 sweep 状态。

    作为存活信号：即使 runner 被一个长时间的 BoTorch 调用阻塞，
    heartbeat 也会持续输出，让操作者知道 sweep 进程还活着（以及
    上次写入数据是多久前）。通过上下文管理器协议启动/停止。

    状态回调返回一行状态文本；回调中的异常会被吞掉，防止格式化
    bug 杀死整个 sweep。
    """

    def __init__(
        self,
        get_state: Callable[[], str],
        interval_s: float = 60.0,
        stream_name: str = "heartbeat",
    ) -> None:
        self._get_state = get_state
        self._interval_s = float(interval_s)
        self._stream_name = stream_name
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def __enter__(self) -> "_Heartbeat":
        if self._interval_s > 0:
            self._thread = threading.Thread(
                target=self._loop, name="benchmark-heartbeat", daemon=True
            )
            self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _loop(self) -> None:
        while not self._stop.wait(self._interval_s):
            try:
                msg = self._get_state()
            except Exception as e:
                msg = f"<state callback failed: {type(e).__name__}: {e}>"
            print(f"[{self._stream_name}] {msg}", flush=True)

# ── 种子推导 ──────────────────────────────────────────────────────────────
# 所有 BenchmarkRunner 级别的种子都从 (strategy, case_id, repeat) 的
# 稳定哈希中推导，这样：
#   * 不同的 case / repeat 在统计上相互独立；
#   * 在同一个 (case, repeat) 内，所有方法共享相同的种子，
#     保持 Friedman / Wilcoxon 符号秩检验所需的*配对样本*结构
#     （Demšar 2006）。
#
# 旧的 ``seed = seed_start + r`` 策略为了历史运行的可复现性而保留。


def derive_seed(
    case_id: str,
    repeat: int,
    *,
    strategy: str = "case_repeat",
    seed_start: int = 1000,
) -> int:
    """为一个 (case, repeat) 单元返回一个非负 31-bit 种子。

    Parameters
    ----------
    case_id : str
        Case 标识符；参与哈希，使不同 case 在同一 repeat 下独立。
    repeat : int
        0-indexed 的重复计数。
    strategy : 'case_repeat' | 'legacy'
        ``case_repeat``（推荐）返回 ``(case_id, repeat, seed_start)``
        的稳定哈希；``legacy`` 返回 ``seed_start + repeat``（与
        v0.3.x 及更早版本相同）。
    seed_start : int
        盐值/起始偏移；默认 1000，与历史运行一致。

    Returns
    -------
    int
        ``[0, 2**31)`` 范围内的非负整数。
    """
    if strategy == "legacy":
        return int(seed_start + repeat)
    if strategy != "case_repeat":
        raise ValueError(f"unknown seed strategy {strategy!r}")
    payload = f"{seed_start}|{case_id}|{repeat}".encode()
    digest = hashlib.blake2b(payload, digest_size=4).digest()
    return int.from_bytes(digest, byteorder="big", signed=False) & 0x7FFFFFFF


# ─── Sampler（采样器）──────────────────────────────────────────────────────
class Sampler(ABC):
    """在归一化的 [0,1]^d 超立方体中生成设计矩阵。"""

    name: str = "<abstract>"

    @abstractmethod
    def sample(self, n: int, d: int, seed: Optional[int] = None) -> np.ndarray:
        """返回形状为 (n, d) 的数组，元素在 [0, 1] 范围内。"""


# ─── Surrogate（代理模型）──────────────────────────────────────────────────
class Surrogate(ABC):
    """拟合回归模型并预测均值（+ 可选标准差）。

    代理模型是*输出无关的*：接受形状为 ``(n, m)`` 的 ``Y_train``，
    返回形状为 ``(n_test, m)`` 的预测。大多数 sklearn 风格的模型
    可以通过为每个输出列训练一个模型来简单包装。
    """

    name: str = "<abstract>"

    @abstractmethod
    def fit(self, X_train: np.ndarray, Y_train: np.ndarray) -> "Surrogate":
        ...

    @abstractmethod
    def predict(self, X_test: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """返回 (均值, 标准差)。如果不支持标准差，``std`` 可以为 ``None``。"""

    def get_init_kwargs(self) -> dict:
        """返回构造函数参数，使代理模型可以被重新实例化。

        带有超参数的子类必须覆盖此方法；默认返回空字典（无超参数）。
        """
        return {}


# ─── BenchmarkMethod（基准测试方法）────────────────────────────────────────
class BenchmarkMethod(ABC):
    """端到端的 DOE 方法：设计 +（可选）代理模型拟合 + 预测。

    实现必须满足：
      * **跨调用无状态** — 每次 ``run`` 调用创建新模型，由 ``seed`` 初始化；
      * **给定种子时确定性** — 相同 ``(case, n_train, X_test, seed)`` 的两次调用
        返回相同预测（除数值噪声外）。

    类属性
    ------
    parallel_backend : 'loky' | 'threading' | 'serial'，默认 'loky'
        在 :class:`BenchmarkRunner` 中 ``n_jobs > 1`` 时的首选并行调度后端。

        * ``'loky'``（默认）— 基于进程；对纯 NumPy / scikit-learn 方法安全且快速。
        * ``'threading'`` — 适用于热循环在 C/C++ 中且*释放 GIL*、但在 Windows
          ``multiprocessing.spawn`` 下**不安全**的方法。BoTorch / GPyTorch 属于
          此类：它们的 MKL 内核操作在 loky 子进程中会 segfault，但在线程间
          可以正确并行。
        * ``'serial'`` — 即使 ``n_jobs > 1`` 也固定为单线程执行；作为
          在其他两种后端下都崩溃或死锁的方法的最后逃生口。

    memory_class : 'light' | 'medium' | 'heavy'，默认 'light'
        粗粒度的内存占用提示，:class:`BenchmarkRunner` 据此决定为此方法
        分配多少 worker。

        * ``'light'``（默认）— sklearn / NumPy 方法，每 worker 峰值 RSS < 200 MB。
          以完整 ``n_jobs`` 运行。
        * ``'medium'`` — 顺序主动学习方法（EGO, MaxVariance, AK-MCS, BoTorch qEI），
          每 worker 峰值 RSS 约 200–500 MB（因为它们反复拟合 GP）。
        * ``'heavy'`` — 如 BoTorch SAASBO 等方法，NUTS 采样器每次拟合分配
          约 1–2 GB。在 threading 后端中运行时，heavy worker 共享父进程 RSS，
          因此 runner 通过 ``bo_n_jobs`` 限制并发数。

        调度策略详见 :class:`BenchmarkRunner`。
    """

    name: str = "<abstract>"
    family: str = "<abstract>"   # 'space_filling+gp', 'sequential_ego' 等
    parallel_backend: str = "loky"
    memory_class: str = "light"   # 'light' | 'medium' | 'heavy'

    @abstractmethod
    def run(
        self,
        case: db.BaseCase,
        n_train: int,
        X_test: np.ndarray,
        seed: int,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], dict]:
        """端到端运行，返回 ``X_test`` 上的预测。

        Returns
        -------
        Y_pred : np.ndarray, 形状 (n_test, m)
            ``X_test`` 上的预测均值。
        Y_std : np.ndarray 或 None, 形状 (n_test, m)
            预测标准差（不确定性）；不可用时为 ``None``。
        info : dict
            诊断信息（训练时间、超参数等）。
        """


# ─── StaticDesignMethod（静态设计方法 = 采样器 + 代理模型组合）──────────
class StaticDesignMethod(BenchmarkMethod):
    """主流的 DOE 流程：采样 N 个点，拟合代理模型，预测。"""

    family = "static_design"

    def __init__(
        self,
        sampler: Sampler,
        surrogate: Surrogate,
        name: Optional[str] = None,
    ) -> None:
        self.sampler = sampler
        self.surrogate = surrogate
        self.name = name or f"{sampler.name}+{surrogate.name}"

    def run(
        self,
        case: db.BaseCase,
        n_train: int,
        X_test: np.ndarray,
        seed: int,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], dict]:
        d = case.meta.dim_input

        t0 = time.perf_counter()
        X_train = self.sampler.sample(n_train, d, seed=seed)
        t_sample = time.perf_counter() - t0

        t0 = time.perf_counter()
        Y_train = case.evaluate(X_train)
        t_eval = time.perf_counter() - t0

        # 跳过含 NaN 的行（评估失败的点）
        mask = ~np.any(np.isnan(Y_train), axis=1)
        if mask.sum() < 2:
            raise RuntimeError(
                f"[{case.meta.case_id}] only {mask.sum()} valid training samples "
                f"out of {n_train}; cannot fit surrogate."
            )
        X_train, Y_train = X_train[mask], Y_train[mask]

        t0 = time.perf_counter()
        # 每次调用创建新的代理模型副本，避免状态泄漏。
        surrogate = self.surrogate.__class__(**self.surrogate.get_init_kwargs())
        surrogate.fit(X_train, Y_train)
        t_fit = time.perf_counter() - t0

        t0 = time.perf_counter()
        Y_pred, Y_std = surrogate.predict(X_test)
        t_pred = time.perf_counter() - t0

        info = {
            "n_train_effective": int(mask.sum()),
            "n_train_failed": int((~mask).sum()),
            "t_sample_s": t_sample,
            "t_eval_train_s": t_eval,
            "t_fit_s": t_fit,
            "t_predict_s": t_pred,
        }
        return Y_pred, Y_std, info


# ─── 结果容器 ────────────────────────────────────────────────────────────
@dataclass
class BenchmarkResult:
    """整洁的结果对象，以 (case_id, method, budget, repeat) 为索引。

    底层由 ``pandas.DataFrame`` 支撑，方便过滤/透视/导出。

    追加性能
    --------
    行数据缓存在 Python 列表中，只在读取时（通过 :meth:`flush` 或
    :class:`BenchmarkRunner` 完成时）才物化为 ``self.df``。这避免了
    旧的 ``O(n^2)`` 分配模式（每追加一行都要对增长的 DataFrame 执行
    ``pd.concat``）。对于 76,800 行的 sweep，节省约 150 GB 的
    总分配压力。

    ``runner.run()`` 返回后读取 ``result.df`` 与之前完全一致：
    runner 会在返回前自动调用 :meth:`flush`。
    """

    df: pd.DataFrame = field(default_factory=pd.DataFrame)
    # 内部缓冲区；行以 O(1) 追加，flush() 时物化。
    _pending: List[dict] = field(default_factory=list, init=False, repr=False)

    def append(self, row: dict) -> None:
        """缓冲一行。O(1)。读取 ``self.df`` 前需调用 :meth:`flush`。"""
        self._pending.append(row)

    def flush(self) -> None:
        """将缓冲行物化到 ``self.df``。

        幂等操作。``BenchmarkRunner.run`` 会在返回前自动调用；
        用户代码只需在运行中检查 ``BenchmarkResult`` 时手动调用
        （例如在线程中）。
        """
        if not self._pending:
            return
        new = pd.DataFrame(self._pending)
        if self.df.empty:
            self.df = new
        else:
            # 合并 schema，保留后到达的列。
            cols = list(dict.fromkeys(list(self.df.columns) + list(new.columns)))
            self.df = pd.concat(
                [self.df.reindex(columns=cols), new.reindex(columns=cols)],
                ignore_index=True,
            )
        self._pending.clear()

    def to_csv(self, path) -> None:
        self.flush()
        self.df.to_csv(path, index=False)

    def to_parquet(self, path) -> None:
        self.flush()
        self.df.to_parquet(path, index=False)

    @classmethod
    def from_csv(cls, path) -> "BenchmarkResult":
        return cls(df=pd.read_csv(path))

    def aggregate(self, by: Sequence[str] = ("case_id", "method", "budget_mult")) -> pd.DataFrame:
        """跨 repeat 聚合：每个数值指标的中位数 + 95% CI。"""
        self.flush()
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        # 排除索引类列
        skip = {
            "repeat",
            "seed",
            "test_seed",
            "n_test",
            "n_train_effective",
            "n_train_failed",
        }
        agg_cols = [c for c in numeric_cols if c not in skip]

        def _agg(g):
            out = {}
            for c in agg_cols:
                vals = g[c].dropna().to_numpy()
                if len(vals) == 0:
                    out[f"{c}_median"] = np.nan
                    out[f"{c}_q025"] = np.nan
                    out[f"{c}_q975"] = np.nan
                    continue
                out[f"{c}_median"] = float(np.median(vals))
                out[f"{c}_q025"] = float(np.quantile(vals, 0.025))
                out[f"{c}_q975"] = float(np.quantile(vals, 0.975))
            out["n_repeats"] = len(g)
            return pd.Series(out)

        return self.df.groupby(list(by), as_index=False).apply(_agg, include_groups=False)


# ─── 单元格执行（顶层函数，便于 joblib pickle）────────────────────────────
def _run_one_cell(
    *,
    case_id: str,
    method: "BenchmarkMethod",
    budget_mult: int,
    repeat: int,
    seed: int,
    seed_strategy: str,
    seed_start: int,
    test_seed: int,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    case_loader,
    case_meta: dict,
    method_meta: dict,
    timeout_s: Optional[float] = None,
) -> dict:
    """执行一个 *(case, method, budget, repeat)* 单元并返回其行数据。

    定义在模块顶层以便 ``joblib.Parallel``（loky 后端）可以 pickle。
    镜像了 :meth:`BenchmarkRunner.run` 中原始顺序内循环的逻辑。

    Parameters
    ----------
    timeout_s : float 或 None
        单元格软超时上限（秒）。设置后，``method.run`` 被派发到
        单线程 :class:`ThreadPoolExecutor`，并用此超时调用
        :meth:`Future.result`。超时时返回 ``status='timeout'`` 的行；
        底层线程*不会*被杀死（Python 缺少可移植的线程取消原语），
        但 runner 会立即进入下一个单元格。
        ``None``（默认）保留无超时的旧行为。
    """
    case = case_loader(case_id)
    d = int(case_meta["dim"])
    n_train = int(budget_mult * d)
    row: dict = {
        "case_id": case_id,
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "case_source_path": case_meta.get("case_source_path", ""),
        "case_source_sha256": case_meta.get("case_source_sha256", ""),
        "domain": case_meta["domain"],
        "dim": d,
        "n_output": case_meta["n_output"],
        "method": method.name,
        "method_family": method.family,
        "method_source_path": method_meta.get("method_source_path", ""),
        "method_source_sha256": method_meta.get("method_source_sha256", ""),
        "budget_mult": int(budget_mult),
        "n_train": n_train,
        "n_test": int(X_test.shape[0]),
        "repeat": int(repeat),
        "seed": int(seed),
        "seed_strategy": str(seed_strategy),
        "seed_start": int(seed_start),
        "test_seed": int(test_seed),
        "worker_host": socket.gethostname(),
        "worker_pid": int(os.getpid()),
        "cell_started_utc": _utc_now_iso(),
        "rss_gb_start": _process_rss_gb(),
        "status": "failed",
        "failure_category": "exception",
        "error": "",
    }
    t0 = time.perf_counter()
    with _RssPeakSampler() as rss_peak:
        try:
            if timeout_s is not None and timeout_s > 0:
                from concurrent.futures import (
                    ThreadPoolExecutor,
                    TimeoutError as _FutTimeout,
                )
                ex = ThreadPoolExecutor(max_workers=1)
                try:
                    fut = ex.submit(method.run, case, n_train, X_test, seed)
                    try:
                        Y_pred, Y_std, info = fut.result(timeout=float(timeout_s))
                    except _FutTimeout:
                        row["t_total_s"] = time.perf_counter() - t0
                        row["status"] = "timeout"
                        row["error"] = (
                            f"timeout after {timeout_s:.0f}s "
                            f"({method.name} on {case_id}, n_train={n_train})"
                        )
                        row["failure_category"] = classify_failure_category(
                            row["status"], row["error"]
                        )
                        row["cell_finished_utc"] = _utc_now_iso()
                        row["rss_gb_end"] = _process_rss_gb()
                        row["rss_gb_peak"] = rss_peak.observe()
                        return row
                finally:
                    # 不阻塞等待挂起的线程释放。
                    ex.shutdown(wait=False)
            else:
                Y_pred, Y_std, info = method.run(case, n_train, X_test, seed)
            row["t_total_s"] = time.perf_counter() - t0
            _merge_method_info(row, info)

            metrics = compute_all_metrics(
                Y_true=Y_test,
                Y_pred=Y_pred,
                Y_std=Y_std,
            )
            row.update(metrics)

            try:
                from benchmark.reliability import reliability_metrics
                rel = reliability_metrics(case_id, Y_test, Y_pred, Y_std)
                row.update(rel)
            except Exception:
                pass

            row["status"] = "ok"
            row["failure_category"] = "ok"
        except Exception as exc:
            row["t_total_s"] = time.perf_counter() - t0
            row["status"] = "failed"
            row["error"] = f"{type(exc).__name__}: {exc}"
            row["failure_category"] = classify_failure_category(
                row["status"], row["error"]
            )
        finally:
            row.setdefault("cell_finished_utc", _utc_now_iso())
            row.setdefault("rss_gb_end", _process_rss_gb())
            row.setdefault("rss_gb_peak", rss_peak.observe())
            row["protocol_digest"] = protocol_digest_for_row(row)
    return row


# ─── BenchmarkRunner（基准测试运行器）────────────────────────────────────
class BenchmarkRunner:
    """编排完整的 *cases × methods × budgets × repeats* sweep。

    Parameters
    ----------
    case_ids : str 序列
        来自 ``doe_testbed.list_cases`` 的 case 标识符。
    methods : BenchmarkMethod 序列
        要比较的方法。
    budgets : int 序列
        训练集大小，以输入维度 ``d`` 的**倍数**表示
        （所以 ``budgets=[5, 10]`` 会运行 ``n_train = 5d`` 和 ``10d``）。
    n_repeats : int
        每个 (case, method, budget) 的独立随机种子数量。
    n_test : int
        参考测试集大小。
    test_seed : int
        参考测试集的固定种子（跨方法/repeat 共享）。
    seed_strategy : 'case_repeat' | 'legacy'
        单元格种子的推导方式；见 :func:`derive_seed`。
        默认 ``'case_repeat'``（推荐用于新运行）。
    seed_start : int
        种子推导的盐值/起始偏移。
    output_dir : Path 或 None
        如果指定，将增量 CSV 行写入该目录，实现崩溃恢复。
    verbose : bool
    n_jobs : int，默认 1
        并行 worker 数量（joblib loky 后端）。``1`` 为顺序运行。
        大于 1 时，每个 case 内的 *(method, budget, repeat)* 三元组
        并行调度；参考测试集的构建在 case 内保持顺序。
    skip_existing : bool，默认 False
        如果为 True 且 ``output_dir/results.csv`` 存在，跳过文件中
        已有 ``(case_id, method, budget_mult, repeat)`` 元组的单元格。
        用于恢复中断的运行或在重新配置后补充现有 CSV。
    timeout_s : float 或 None，默认 None
        对 ``parallel_backend`` 不是 ``'threading'`` 的方法施加的
        单元格软超时（秒）。超过时记录 ``status='timeout'`` 的行
        并继续下一个任务。``None`` 禁用超时。
    bo_timeout_s : float 或 None，默认 None
        对声明 ``parallel_backend='threading'`` 的方法（BoTorch / GPyTorch）
        的单元格超时；这些方法通常比 scikit-learn 方法慢很多，
        所以有独立的（更大的）超时。``None`` 回退到 ``timeout_s``。
    ref_timeout_s : float 或 None，默认 600
        参考测试集构建（``case.sample`` + ``case.evaluate``）的软超时。
        工程 case 偶尔会在 FEM / 多物理场求解器中挂起；没有超时的话，
        一个坏 case 会冻结整个 sweep。超时时跳过该 case 并记录警告。
    watchdog_s : float 或 None，默认 None
        如果设置，当并行调度生成器在 ``watchdog_s`` 秒内未 yield 新行时
        抛出 :class:`RuntimeError`。runner 捕获后回退到串行模式。
        推荐设置：``2 * max(timeout_s, bo_timeout_s) + 60``。
        ``None``（默认）禁用 watchdog。
    heartbeat_s : float 或 None，默认 60
        存活信号的发射间隔（秒）。格式为
        ``[heartbeat] N/T rows  ...  last row Xs ago``。
        ``None`` 或 ``0`` 禁用心跳。
    max_wall_clock_s : float 或 None，默认 None
        ``run()`` 的总挂钟时间预算。超时后 runner 停止调度新分区，
        返回已写入磁盘的结果。``None`` 表示无上限。
    bo_n_jobs : int 或 None，默认 None
        ``memory_class == 'heavy'`` 的方法（通常是 BoTorch SAASBO）
        使用的并行 worker 数量覆盖。``None``（默认）时 runner 使用
        ``max(1, n_jobs // 4)``——保守的限制，在 16 GB 主机上保持
        总 RSS 在 ``n_jobs * ~1 GB`` 以内。设为 ``1`` 为最保守，
        在大内存服务器上可设更高。
    mem_cap_gb : float 或 None，默认 None
        软 RSS 上限。当父进程超过此 GB 数时，runner 对下一个分区
        降为 ``n_jobs == 1`` 并运行 ``gc.collect()``。需要安装
        ``psutil``；未安装时上限被忽略。``None`` 禁用此保护。
    """

    def __init__(
        self,
        case_ids: Sequence[str],
        methods: Sequence[BenchmarkMethod],
        budgets: Sequence[int],
        n_repeats: int = 10,
        n_test: int = 2000,
        test_seed: int = 0,
        seed_start: int = 1000,
        seed_strategy: str = "case_repeat",
        output_dir: Optional[Path] = None,
        verbose: bool = True,
        case_loader=None,
        n_jobs: int = 1,
        skip_existing: bool = False,
        timeout_s: Optional[float] = None,
        bo_timeout_s: Optional[float] = None,
        ref_timeout_s: Optional[float] = 600.0,
        watchdog_s: Optional[float] = None,
        heartbeat_s: Optional[float] = 60.0,
        max_wall_clock_s: Optional[float] = None,
        bo_n_jobs: Optional[int] = None,
        mem_cap_gb: Optional[float] = None,
    ) -> None:
        self.case_ids = list(case_ids)
        self.methods = list(methods)
        self.budgets = list(budgets)
        self.n_repeats = int(n_repeats)
        self.n_test = int(n_test)
        self.test_seed = int(test_seed)
        self.seed_start = int(seed_start)
        if seed_strategy not in {"case_repeat", "legacy"}:
            raise ValueError(
                f"seed_strategy must be 'case_repeat' or 'legacy', got {seed_strategy!r}"
            )
        self.seed_strategy = seed_strategy
        self.output_dir = Path(output_dir) if output_dir else None
        self.verbose = bool(verbose)
        # 自定义 case 加载器（用于解析函数）；默认为 db.get_case
        self.case_loader = case_loader if case_loader is not None else db.get_case
        self.n_jobs = int(n_jobs)
        self.skip_existing = bool(skip_existing)
        self.timeout_s = float(timeout_s) if timeout_s else None
        self.bo_timeout_s = float(bo_timeout_s) if bo_timeout_s else None
        self.ref_timeout_s = float(ref_timeout_s) if ref_timeout_s else None
        self.watchdog_s = float(watchdog_s) if watchdog_s else None
        self.heartbeat_s = float(heartbeat_s) if heartbeat_s else None
        self.max_wall_clock_s = (
            float(max_wall_clock_s) if max_wall_clock_s else None
        )
        # Heavy 方法（BoTorch SAASBO）的并行度上限。默认
        # max(1, n_jobs // 4)：SAASBO 持有约 1-2 GB 常驻内存，
        # 所以 8-job 机器上 4 个同时运行的实例 = 4-8 GB heavy 份额，
        # 为 loky 池中的 light 方法留出空间。
        if bo_n_jobs is None:
            self.bo_n_jobs = max(1, self.n_jobs // 4)
        else:
            self.bo_n_jobs = max(1, int(bo_n_jobs))
        self.mem_cap_gb = float(mem_cap_gb) if mem_cap_gb else None

        # 如果操作者要求内存上限但 psutil 不可导入，大声警告：
        # 否则会静默跳过上限，历史上这导致了本应被捕获的 OOM。
        if self.mem_cap_gb is not None:
            try:
                import psutil  # type: ignore  # noqa: F401
            except Exception:
                warnings.warn(
                    f"--mem-cap-gb={self.mem_cap_gb} requested but "
                    f"`psutil` is not installed; the soft RSS guard "
                    f"will NOT activate.  Install with `pip install "
                    f"psutil>=5.9` for the cap to take effect."
                )

        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def _run_config_payload(self, *, total_cells: int) -> dict[str, Any]:
        """Return a durable sweep configuration snapshot for server archives."""
        cases = []
        for case_id in self.case_ids:
            case = self.case_loader(case_id)
            meta = case.meta
            cases.append({
                "case_id": str(case_id),
                "domain": str(meta.domain),
                "dim": int(meta.dim_input),
                "n_output": int(meta.dim_output),
                **case_source_fingerprint(case),
            })

        methods = []
        for method in self.methods:
            methods.append({
                "method": str(method.name),
                "method_family": str(method.family),
                "parallel_backend": str(getattr(method, "parallel_backend", "loky") or "loky"),
                "memory_class": str(getattr(method, "memory_class", "light") or "light"),
                **method_source_fingerprint(method),
            })

        return {
            "artifact": "doe-benchmark-stage2-run-config",
            "created_utc": _utc_now_iso(),
            "result_schema_version": int(RESULT_SCHEMA_VERSION),
            "total_cells": int(total_cells),
            "cases": cases,
            "methods": methods,
            "budgets": [int(b) for b in self.budgets],
            "n_repeats": int(self.n_repeats),
            "n_test": int(self.n_test),
            "test_seed": int(self.test_seed),
            "seed_start": int(self.seed_start),
            "seed_strategy": str(self.seed_strategy),
            "n_jobs": int(self.n_jobs),
            "bo_n_jobs": int(self.bo_n_jobs),
            "timeout_s": self.timeout_s,
            "bo_timeout_s": self.bo_timeout_s,
            "ref_timeout_s": self.ref_timeout_s,
            "watchdog_s": self.watchdog_s,
            "heartbeat_s": self.heartbeat_s,
            "max_wall_clock_s": self.max_wall_clock_s,
            "mem_cap_gb": self.mem_cap_gb,
            "skip_existing": bool(self.skip_existing),
            "output_dir": str(self.output_dir) if self.output_dir is not None else None,
            "torch_reproducibility": _torch_reproducibility_snapshot(
                apply_policy=True
            ),
            "environment": _runtime_environment_snapshot(),
            "temp_storage": _temp_storage_snapshot(),
            "thread_env": {
                name: os.environ.get(name)
                for name in (
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                )
            },
        }

    def _default_watchdog_s(self) -> Optional[float]:
        """根据单元格超时计算默认的生成器卡死上限。"""
        if self.watchdog_s is not None:
            return self.watchdog_s
        cell_caps = [t for t in (self.timeout_s, self.bo_timeout_s) if t]
        if not cell_caps:
            return None
        # 2 倍最慢单元格 + 60 秒缓冲，覆盖 worker 重启和排队。
        return 2.0 * max(cell_caps) + 60.0

    @staticmethod
    def _effective_backend(method: "BenchmarkMethod", n_jobs: int) -> str:
        """返回实际使用的调度后端。

        优先使用 :attr:`BenchmarkMethod.parallel_backend`，
        但 ``n_jobs <= 1`` 时降级为 ``'serial'``，保持纯静态方法
        路径的单线程确定性。
        """
        declared = getattr(method, "parallel_backend", "loky") or "loky"
        if n_jobs <= 1:
            return "serial"
        if declared not in {"loky", "threading", "serial"}:
            return "loky"
        return declared

    def _n_jobs_for_class(self, memory_class: str) -> int:
        """返回指定内存类别的 worker 数量。

        * ``'heavy'`` → ``self.bo_n_jobs``（默认 ``max(1, n_jobs // 4)``）。
          上限存在是因为 BoTorch SAASBO 在 threading 后端的共享进程中
          每实例持有约 1–2 GB；8 个同时运行的实例会在 16 GB Windows
          主机上 OOM。
        * ``'medium'`` → ``max(1, n_jobs // 2)``，用于顺序 GP 拟合方法
          （每实例约 200–500 MB）。半速保持总 RSS 可控的同时仍然并行。
        * ``'light'`` → ``self.n_jobs``（全速用于仅 sklearn 的方法）。
        """
        if memory_class == "heavy":
            return self.bo_n_jobs
        if memory_class == "medium":
            return max(1, self.n_jobs // 2)
        return self.n_jobs

    def _timeout_for(self, method: "BenchmarkMethod") -> Optional[float]:
        """返回适用于 *method* 的单元格超时。"""
        backend = getattr(method, "parallel_backend", "loky") or "loky"
        if backend == "threading" and self.bo_timeout_s is not None:
            return self.bo_timeout_s
        return self.timeout_s

    @staticmethod
    def _rss_gb() -> Optional[float]:
        """返回当前进程 RSS（GB），psutil 不可用时返回 ``None``。"""
        try:
            import psutil  # type: ignore
            return psutil.Process().memory_info().rss / (1024 ** 3)
        except Exception:
            return None

    def run(self) -> BenchmarkResult:
        """执行完整的 sweep。返回 :class:`BenchmarkResult`。

        调度策略
        --------
        在每个 case 内，*(method, budget, repeat)* 三元组按
        :meth:`BenchmarkMethod.parallel_backend` 分区调度：

        * 声明 ``'loky'``（默认）的方法首先通过 joblib 的 ``loky`` 进程池
          调度——对 sklearn / NumPy 方法最快；
        * 声明 ``'threading'`` 的方法接下来通过 joblib 的 ``threading`` 池
          调度——BoTorch / GPyTorch 在 Windows 上需要，因为 MKL 内核操作
          在 spawn 子进程中会 segfault；
        * 声明 ``'serial'`` 的方法（或 ``self.n_jobs <= 1`` 时的任何分区）
          在主线程上运行。

        卡死防御
        --------
        每个并行分区被三层防御机制包裹：

        1. *单元格软超时*（``self.timeout_s`` / ``self.bo_timeout_s``）
           在 :func:`_run_one_cell` 内生效——产生 ``status='timeout'`` 的行。
        2. *生成器卡死 watchdog*（``self.watchdog_s`` 或从单元格超时推导的值）：
           如果 joblib 的生成器长时间沉默，分区中止，剩余任务回退到串行。
        3. *Worker 死亡恢复*（捕获 :class:`TerminatedWorkerError` 及其他
           调度错误）：分区的未完成任务串行重跑。

        参考测试集构建（``case.sample`` + ``case.evaluate``）本身也被
        :func:`_call_with_soft_timeout` 包裹，上限为 ``self.ref_timeout_s``——
        一个挂起的物理求解器因此会跳过其 case 而不是冻结整个 sweep。

        ``self.heartbeat_s``（默认 60 秒）启动一个守护线程，定期输出
        ``[heartbeat] N/T rows ... last row Xs ago``，让操作者在漫长的
        BoTorch / SAASBO 单元格期间有存活信号。

        ``self.max_wall_clock_s`` 限制总挂钟时间；超时后 runner 停止调度
        新分区，返回已写入的结果。

        ``self.skip_existing`` 从 ``output_dir/results.csv`` 恢复。
        """
        result = BenchmarkResult()
        total = (
            len(self.case_ids) * len(self.methods) * len(self.budgets) * self.n_repeats
        )
        # 使用 ``time.monotonic`` 进行预算计算，这样系统时钟跳变
        # （NTP 校正、夏令时、容器迁移）不会触发挂钟上限或心跳
        # 过期检测。（``time.time()`` 会受影响。）
        run_started_at = time.monotonic()

        # 构建 skip 集合（如果 --skip-existing）
        skip_set: Set[Tuple[str, str, int, int]] = set()
        if self.skip_existing and self.output_dir is not None:
            csv_path = self.output_dir / "results.csv"
            if csv_path.is_symlink():
                raise ValueError(
                    "--skip-existing cannot resume from a symlinked results.csv. "
                    f"File: {csv_path}. Use a real file in the output directory "
                    "so the runner cannot read from or append to an external target."
                )
            if csv_path.exists():
                csv_contract_errors = csv_header_contract_errors(csv_path)
                if csv_contract_errors:
                    raise ValueError(
                        "--skip-existing cannot resume from an unreadable or "
                        f"corrupt results.csv. File: {csv_path}. "
                        f"CSV contract errors: {csv_contract_errors}. "
                        "Use a clean output directory, restore a parseable "
                        "archive, or repair the CSV explicitly before resuming."
                    )
                try:
                    existing = pd.read_csv(csv_path)
                except Exception as exc:
                    raise ValueError(
                        "--skip-existing cannot resume from an unreadable or "
                        f"corrupt results.csv. File: {csv_path}. "
                        f"Parser error: {type(exc).__name__}: {exc}. "
                        "Use a clean output directory, restore a parseable "
                        "archive, or repair the CSV explicitly before resuming."
                    )
                if existing.empty:
                    raise ValueError(
                        "--skip-existing cannot resume from a results.csv that "
                        "has a header but no completed data rows. "
                        f"File: {csv_path}. This usually means a stale startup "
                        "file, interrupted writer, or wrong output directory. "
                        "Move it aside, use a clean output directory, or rerun "
                        "without --skip-existing for a fresh sweep."
                    )
                self._validate_resume_grid(existing, csv_path)
                self._validate_resume_run_config(
                    self.output_dir / "run_config.json",
                    csv_path,
                    total_cells=total,
                )
                existing_keys = existing[
                    ["case_id", "method", "budget_mult", "repeat"]
                ]
                skip_set = {
                    (str(c), str(m), int(b), int(r))
                    for c, m, b, r
                    in existing_keys.itertuples(index=False, name=None)
                }
                if self.verbose:
                    print(f"[skip-existing] loaded {len(skip_set):,} cells from "
                          f"{csv_path} — will be skipped.")

        # 锁定 BLAS / OpenMP 线程数为 1，避免多线程 BLAS 过度订阅。
        if self.n_jobs and self.n_jobs > 1:
            for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                        "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
                os.environ.setdefault(var, "1")

        if self.output_dir is not None:
            _atomic_write_json(
                self.output_dir / "run_config.json",
                self._run_config_payload(total_cells=total),
            )

        # 懒加载 joblib（仅并行时需要）
        Parallel = delayed = None
        TerminatedWorkerError = Exception  # 后备，使 `except` 始终可用
        if self.n_jobs and self.n_jobs > 1:
            from joblib import Parallel as _Parallel, delayed as _delayed
            Parallel = _Parallel
            delayed = _delayed
            try:
                from joblib.externals.loky.process_executor import (
                    TerminatedWorkerError as _TWE,
                )
                TerminatedWorkerError = _TWE
            except ImportError:
                TerminatedWorkerError = Exception

        ref_sets: dict = {}
        idx = 0
        last_row_ts = run_started_at
        watchdog_s = self._default_watchdog_s()

        csv_path_out = (
            self.output_dir / "results.csv" if self.output_dir is not None else None
        )
        if csv_path_out is not None and not self.skip_existing and csv_path_out.exists():
            csv_path_out.unlink()

        def _heartbeat_msg() -> str:
            now = time.monotonic()
            elapsed = now - run_started_at
            stale = now - last_row_ts
            todo = max(total - len(skip_set) - idx, 0)
            if idx > 0 and elapsed > 0:
                rate = idx / elapsed  # 本次运行的行/秒
                eta_s = todo / rate if rate > 0 else float("inf")
                eta = f"ETA {eta_s/3600:.1f}h"
            else:
                eta = "ETA --"
            done_pct = 100.0 * (idx + len(skip_set)) / max(total, 1)
            # 可选的 RSS 读数：需要 psutil；缺失时静默跳过。
            rss_gb = self._rss_gb()
            rss_msg = f"  rss {rss_gb:.1f}GB" if rss_gb is not None else ""
            return (
                f"{idx}+{len(skip_set)}/{total} ({done_pct:.1f}%)  "
                f"new {idx}  elapsed {elapsed/60:.1f}m  "
                f"last row {stale:.0f}s ago  {eta}{rss_msg}"
            )

        def _budget_left() -> bool:
            """还有挂钟时间可以调度新工作时返回 True。"""
            if self.max_wall_clock_s is None:
                return True
            return (time.monotonic() - run_started_at) < self.max_wall_clock_s

        def _consume(row, partition_done_keys):
            """将一行数据写入磁盘 + 内存 + 控制台。通过闭包修改 idx/last_row_ts/result。"""
            nonlocal idx, last_row_ts
            idx += 1
            last_row_ts = time.monotonic()
            if row.get("status") in {"failed", "timeout"}:
                warnings.warn(
                    f"[{row.get('case_id')}/{row.get('method')}/"
                    f"n={row.get('n_train','?')}/r={row.get('repeat','?')}] "
                    f"{row.get('status')}: {row.get('error','?')}"
                )
            if self.verbose:
                print(f"  [{idx}/{total}] {row.get('case_id')} | {row.get('method')} "
                      f"| n={row.get('n_train','?')} (={row.get('budget_mult','?')}d) "
                      f"| rep={row.get('repeat','?')} | status={row.get('status')}")
            result.append(row)
            if csv_path_out is not None:
                self._stream_append_row(csv_path_out, row)
            partition_done_keys.add(
                (row.get("case_id"), row.get("method"),
                 int(row.get("budget_mult", -1)), int(row.get("repeat", -1)))
            )

        def _run_partition(partition, backend: str, n_jobs_override: Optional[int] = None):
            """调度一个 (case, backend, 内存类别) 分区，带卡死 + 崩溃恢复。

            Parameters
            ----------
            partition : (method, budget_mult, repeat, seed) 元组列表
            backend : 'loky' | 'threading' | 'serial'
            n_jobs_override : int 或 None
                此分区使用的 worker 数量。``None`` 使用 ``self.n_jobs``。
                runner 在此传入基于内存类别的值，使 heavy 方法（BoTorch SAASBO）
                获得 fewer worker。
            """
            done_keys: set = set()
            if not partition:
                return

            timeout_for_method = {
                id(m): self._timeout_for(m) for m, *_ in partition
            }
            n_jobs_eff = (
                int(n_jobs_override) if n_jobs_override is not None else self.n_jobs
            )

            def _make_call(method, budget_mult, r, seed):
                return dict(
                    case_id=case_id,
                    method=method,
                    budget_mult=budget_mult,
                    repeat=r,
                    seed=seed,
                    seed_strategy=self.seed_strategy,
                    seed_start=self.seed_start,
                    test_seed=self.test_seed,
                    X_test=X_test,
                    Y_test=Y_test,
                    case_loader=self.case_loader,
                    case_meta=case_meta,
                    method_meta=method_source_fingerprint(method),
                    timeout_s=timeout_for_method[id(method)],
                )

            def _drain_serial(remaining_tasks):
                for method, budget_mult, r, seed in remaining_tasks:
                    if not _budget_left():
                        return
                    row = _run_one_cell(**_make_call(method, budget_mult, r, seed))
                    _consume(row, done_keys)

            if backend == "serial" or Parallel is None or n_jobs_eff <= 1:
                _drain_serial(partition)
                return

            # --- 带 watchdog + 崩溃恢复的并行调度 --------
            try:
                gen = Parallel(
                    n_jobs=n_jobs_eff,
                    backend=backend,
                    return_as="generator",
                    verbose=0,
                )(
                    delayed(_run_one_cell)(**_make_call(method, budget_mult, r, seed))
                    for method, budget_mult, r, seed in partition
                )
                for row in _iter_with_watchdog(gen, watchdog_s):
                    _consume(row, done_keys)
                    if not _budget_left():
                        # 挂钟时间在分区内耗尽——停止从 joblib 拉取
                        # （GC 时它会自行清理）。
                        break
            except TerminatedWorkerError as exc:
                warnings.warn(
                    f"[{case_id}/backend={backend}/n_jobs={n_jobs_eff}] worker died "
                    f"({type(exc).__name__}); falling back to serial for "
                    f"remaining tasks. Original error: {exc}"
                )
            except Exception as exc:
                # 捕获我们自己的 watchdog RuntimeError 及任何调度级错误。
                warnings.warn(
                    f"[{case_id}/backend={backend}/n_jobs={n_jobs_eff}] parallel dispatch aborted "
                    f"({type(exc).__name__}: {exc}); falling back to serial."
                )
            else:
                return  # 干净完成——无需回退。

            remaining = [
                (m, b, r, s) for m, b, r, s in partition
                if (case_id, m.name, int(b), int(r)) not in done_keys
            ]
            if self.verbose:
                print(f"  [{case_id}/backend={backend}] "
                      f"recovering {len(remaining)} unfinished tasks serially.")
            _drain_serial(remaining)

        # ── 主循环，逐 case 处理，包裹在心跳线程中 ───────────
        with _Heartbeat(_heartbeat_msg, interval_s=(self.heartbeat_s or 0)):
            for case_id in self.case_ids:
                if not _budget_left():
                    if self.verbose:
                        print(f"[wall-clock] budget exhausted "
                              f"({self.max_wall_clock_s}s); stopping early.")
                    break

                case = self.case_loader(case_id)
                d = case.meta.dim_input
                case_meta = {
                    "domain": case.meta.domain,
                    "dim": d,
                    "n_output": case.meta.dim_output,
                    **case_source_fingerprint(case),
                }

                if case_id not in ref_sets:
                    if self.verbose:
                        print(f"[{case_id}] computing reference set "
                              f"(n={self.n_test}, ref_timeout={self.ref_timeout_s}s)...")
                    ref, reason = _call_with_soft_timeout(
                        self._build_reference_set, case,
                        timeout_s=self.ref_timeout_s,
                    )
                    if reason is not None:
                        warnings.warn(
                            f"[{case_id}] reference-set construction failed "
                            f"({reason}); skipping this case."
                        )
                        ref_sets[case_id] = None  # 标记，不再重试
                        continue
                    ref_sets[case_id] = ref
                if ref_sets[case_id] is None:
                    continue
                X_test, Y_test = ref_sets[case_id]

                # 构建此 case 的 (method, budget, repeat) 任务列表，
                # 按 (有效后端, 内存类别) 双重分区，使每个 (后端, 内存类别)
                # 子分区获得自己的 n_jobs 覆盖。这是防御"8 个同时 SAASBO 实例
                # 导致 OOM"的核心机制：heavy BoTorch 分区以 bo_n_jobs
                # （如 1-2）调度，而 light sklearn 方法保持完整 n_jobs。
                partitions: dict = {}
                for method in self.methods:
                    eff_backend = self._effective_backend(method, self.n_jobs)
                    mem_class = getattr(method, "memory_class", "light") or "light"
                    if mem_class not in {"light", "medium", "heavy"}:
                        mem_class = "light"
                    key_part = (eff_backend, mem_class)
                    partitions.setdefault(key_part, [])
                    for budget_mult in self.budgets:
                        for r in range(self.n_repeats):
                            key = (case_id, method.name, int(budget_mult), int(r))
                            if key in skip_set:
                                continue
                            seed = derive_seed(
                                case_id, r,
                                strategy=self.seed_strategy,
                                seed_start=self.seed_start,
                            )
                            partitions[key_part].append(
                                (method, int(budget_mult), int(r), int(seed))
                            )

                n_in_case = sum(len(p) for p in partitions.values())
                n_skipped_this_case = (
                    len(self.methods) * len(self.budgets) * self.n_repeats - n_in_case
                )
                if self.verbose:
                    summary = ", ".join(
                        f"{b}/{m}={len(p)}" for (b, m), p in partitions.items() if p
                    )
                    rss = self._rss_gb()
                    rss_msg = f"  rss={rss:.1f}GB" if rss is not None else ""
                    print(f"[{case_id}] tasks={n_in_case} ({summary or 'none'})  "
                          f"skipped={n_skipped_this_case}  "
                          f"n_jobs(light/med/heavy)="
                          f"{self.n_jobs}/{max(1, self.n_jobs // 2)}/{self.bo_n_jobs}  "
                          f"watchdog={watchdog_s}s{rss_msg}")

                if not n_in_case:
                    # 释放参考测试集再进入下一个 case；此处无需计算。
                    ref_sets[case_id] = None
                    continue

                # 软内存上限检查：如果 RSS 已超过上限，
                # 将此 case 降为串行调度并立即 gc。
                forced_serial = False
                if self.mem_cap_gb is not None:
                    rss_now = self._rss_gb()
                    if rss_now is not None and rss_now > self.mem_cap_gb:
                        warnings.warn(
                            f"[{case_id}] RSS {rss_now:.1f}GB > cap "
                            f"{self.mem_cap_gb:.1f}GB — forcing this case to serial "
                            f"and running gc.collect()."
                        )
                        import gc
                        gc.collect()
                        forced_serial = True

                # 按优先级顺序处理分区：
                #   1. light loky      — 完整 n_jobs，最快
                #   2. medium loky     — 半 n_jobs，GP 拟合开销受控
                #   3. heavy loky      — bo_n_jobs（罕见；loky 安全的 BO）
                #   4. light threading — 完整 n_jobs（不常见）
                #   5. medium threading — 半 n_jobs
                #   6. heavy threading — bo_n_jobs（BoTorch SAASBO 主要场景）
                #   7. serial          — 单线程收尾
                priority_order = [
                    ("loky", "light"), ("loky", "medium"), ("loky", "heavy"),
                    ("threading", "light"), ("threading", "medium"), ("threading", "heavy"),
                    ("serial", "light"), ("serial", "medium"), ("serial", "heavy"),
                ]
                for (backend, mem_class) in priority_order:
                    if not _budget_left():
                        break
                    part = partitions.get((backend, mem_class), [])
                    if not part:
                        continue
                    n_jobs_for_part = (
                        1 if forced_serial else self._n_jobs_for_class(mem_class)
                    )
                    _run_partition(part, backend, n_jobs_override=n_jobs_for_part)

                # 释放此 case 的参考测试集张量（不会再访问它）。
                # 这将每次运行的内存增长限制为一个 case 的 X_test/Y_test 数组，
                # 而不是跨 case 累积。
                ref_sets[case_id] = None
                # 释放 joblib loky worker，清理每 worker 的缓存。
                # 这在长时间 sweep 中很关键——worker 会缓慢积累 sklearn
                # 警告状态、numpy 临时缓冲区和 torch 张量，直到 worker 退出
                # 才归还给操作系统。
                try:
                    from joblib.externals.loky import (  # type: ignore
                        get_reusable_executor,
                    )
                    get_reusable_executor().shutdown(kill_workers=True)
                except Exception:
                    pass
                # 强制 GC；父进程否则会保持 run 循环的闭包和中间 joblib
                # 生成器存活。
                import gc
                gc.collect()

        # 返回前物化缓冲行，使用户代码读取 ``result.df`` 时
        # 能立即看到完整的 DataFrame。
        result.flush()
        if self.verbose:
            print(f"\nFinished: {len(result.df)} rows total in this run.")
        return result

    def _validate_resume_grid(self, existing: pd.DataFrame, csv_path: Path) -> None:
        """Fail fast if ``--skip-existing`` would resume from another protocol.

        Resumability is only safe when the existing CSV is a subset of the
        current declared grid.  Otherwise rows from an older pool, budget set,
        or repeat count can leak into the final results and corrupt paper
        denominators before the later integrity gate catches it.
        """
        expected_cases = {str(c) for c in self.case_ids}
        expected_methods = {str(m.name) for m in self.methods}
        expected_budgets = {int(b) for b in self.budgets}
        expected_repeats = set(range(self.n_repeats))

        missing_cols = sorted(_RESUME_REQUIRED_COLUMNS - set(existing.columns))
        if missing_cols:
            raise ValueError(
                "--skip-existing cannot resume from a results.csv with a stale "
                "or incomplete schema. "
                f"File: {csv_path}. Missing required columns: {missing_cols}. "
                "Use a clean output directory or regenerate the CSV with the "
                "current runner."
            )

        missing_grid_values: list[str] = []
        for col in ("case_id", "method"):
            text = existing[col].fillna("").astype(str).str.strip()
            n_missing = int(text.eq("").sum())
            if n_missing:
                missing_grid_values.append(f"{col}={n_missing}")
        for col in ("budget_mult", "repeat"):
            n_missing = int(existing[col].isna().sum())
            if n_missing:
                missing_grid_values.append(f"{col}={n_missing}")
        if missing_grid_values:
            raise ValueError(
                "--skip-existing cannot resume from a results.csv with blank "
                "or missing protocol key values. "
                f"File: {csv_path}. Missing values: {missing_grid_values}. "
                "Use a clean output directory or repair the CSV explicitly."
            )

        observed_cases = set(existing["case_id"].dropna().astype(str))
        observed_methods = set(existing["method"].dropna().astype(str))
        problems: list[str] = []
        observed_budgets = _resume_int_set(existing["budget_mult"], "budget_mult", problems)
        observed_repeats = _resume_int_set(existing["repeat"], "repeat", problems)
        extra_cases = sorted(observed_cases - expected_cases)
        extra_methods = sorted(observed_methods - expected_methods)
        extra_budgets = sorted(observed_budgets - expected_budgets)
        extra_repeats = sorted(observed_repeats - expected_repeats)
        if extra_cases:
            problems.append(f"unexpected cases={extra_cases}")
        if extra_methods:
            problems.append(f"unexpected methods={extra_methods}")
        if extra_budgets:
            problems.append(f"unexpected budgets={extra_budgets}")
        if extra_repeats:
            problems.append(f"unexpected repeats={extra_repeats}")

        status_text = existing["status"].astype(str).str.strip()
        bad_status = sorted(set(status_text.dropna()) - _RESUME_ALLOWED_STATUSES)
        if bad_status:
            problems.append(f"invalid status values={bad_status}")
        if existing["status"].isna().any() or status_text.eq("").any():
            problems.append("missing status values")
        failure_category_text = (
            existing["failure_category"].fillna("").astype(str).str.strip()
        )
        bad_failure_category = sorted(
            set(failure_category_text) - ALLOWED_FAILURE_CATEGORIES - {""}
        )
        if bad_failure_category:
            problems.append(f"invalid failure_category values={bad_failure_category}")
        if existing["failure_category"].isna().any() or failure_category_text.eq("").any():
            problems.append("missing failure_category values")

        t_total_s = pd.to_numeric(existing["t_total_s"], errors="coerce")
        bad_timing = t_total_s.isna() | ~np.isfinite(t_total_s) | (t_total_s < 0)
        if bool(bad_timing.any()):
            problems.append(f"invalid t_total_s rows={int(bad_timing.sum())}")

        worker_host = existing["worker_host"].fillna("").astype(str).str.strip()
        if bool(worker_host.eq("").any()):
            problems.append(f"empty worker_host rows={int(worker_host.eq('').sum())}")
        worker_pid, valid_worker_pid = strict_integer_series(existing["worker_pid"])
        bad_pid = ~valid_worker_pid | worker_pid.le(0)
        if bool(bad_pid.any()):
            problems.append(f"invalid worker_pid rows={int(bad_pid.sum())}")
        parsed_times: dict[str, pd.Series] = {}
        for col in ("cell_started_utc", "cell_finished_utc"):
            text = existing[col].fillna("").astype(str).str.strip()
            parsed = pd.to_datetime(text, errors="coerce", utc=True)
            parsed_times[col] = parsed
            bad_ts = text.eq("") | parsed.isna()
            if bool(bad_ts.any()):
                problems.append(f"invalid {col} rows={int(bad_ts.sum())}")
        bad_time_order = (
            parsed_times["cell_finished_utc"] < parsed_times["cell_started_utc"]
        )
        if bool(bad_time_order.any()):
            problems.append(
                "cell_finished_utc before cell_started_utc "
                f"rows={int(bad_time_order.sum())}"
            )
        bad_duration = invalid_runtime_duration_mask(
            existing["t_total_s"],
            parsed_times["cell_started_utc"],
            parsed_times["cell_finished_utc"],
        )
        if bool(bad_duration.any()):
            problems.append(
                "t_total_s inconsistent with cell runtime timestamps "
                f"rows={int(bad_duration.sum())}"
            )
        rss_values: dict[str, pd.Series] = {}
        for col in ("rss_gb_start", "rss_gb_end", "rss_gb_peak"):
            values = pd.to_numeric(existing[col], errors="coerce")
            rss_values[col] = values
            bad_rss = values.isna() | ~np.isfinite(values) | (values < 0)
            if bool(bad_rss.any()):
                problems.append(f"invalid {col} rows={int(bad_rss.sum())}")
        bad_peak_order = (
            rss_values["rss_gb_peak"].notna()
            & np.isfinite(rss_values["rss_gb_peak"])
            & (
                rss_values["rss_gb_peak"].lt(rss_values["rss_gb_start"])
                | rss_values["rss_gb_peak"].lt(rss_values["rss_gb_end"])
            )
        )
        if bool(bad_peak_order.any()):
            problems.append(
                "rss_gb_peak below start/end rows="
                f"{int(bad_peak_order.sum())}"
            )

        ok_mask = status_text.eq("ok")
        non_ok_mask = ~ok_mask
        nrmse = pd.to_numeric(existing["nrmse"], errors="coerce")
        bad_ok_nrmse = ok_mask & (nrmse.isna() | ~np.isfinite(nrmse) | (nrmse < 0))
        if bool(bad_ok_nrmse.any()):
            problems.append(f"invalid ok nrmse rows={int(bad_ok_nrmse.sum())}")
        nrmse_text = existing["nrmse"].fillna("").astype(str).str.strip()
        non_ok_nrmse = non_ok_mask & nrmse_text.ne("")
        if bool(non_ok_nrmse.any()):
            problems.append(
                "timeout/failed rows with non-empty nrmse="
                f"{int(non_ok_nrmse.sum())}"
            )
        for metric in _RESUME_OPTIONAL_OK_NUMERIC_METRICS:
            if metric not in existing.columns:
                continue
            values = pd.to_numeric(existing[metric], errors="coerce")
            bad_metric = ok_mask & (values.isna() | ~np.isfinite(values))
            if metric != "r2":
                bad_metric = bad_metric | (ok_mask & (values < 0))
            if bool(bad_metric.any()):
                problems.append(f"invalid ok {metric} rows={int(bad_metric.sum())}")
            metric_text = existing[metric].fillna("").astype(str).str.strip()
            non_ok_metric = non_ok_mask & metric_text.ne("")
            if bool(non_ok_metric.any()):
                problems.append(
                    f"timeout/failed rows with non-empty {metric}="
                    f"{int(non_ok_metric.sum())}"
                )
        error_text = existing["error"].fillna("").astype(str).str.strip()
        ok_with_error = ok_mask & error_text.ne("")
        if bool(ok_with_error.any()):
            problems.append(f"ok rows with non-empty error={int(ok_with_error.sum())}")
        non_ok_missing_error = status_text.isin({"timeout", "failed"}) & error_text.eq("")
        if bool(non_ok_missing_error.any()):
            problems.append(
                f"timeout/failed rows with empty error={int(non_ok_missing_error.sum())}"
            )
        bad_failure_category_contract = pd.Series(False, index=existing.index)
        for idx in existing.index:
            if not failure_category_is_consistent(
                existing.at[idx, "status"],
                existing.at[idx, "error"],
                existing.at[idx, "failure_category"],
            ):
                bad_failure_category_contract.loc[idx] = True
        if bool(bad_failure_category_contract.any()):
            problems.append(
                "failure_category inconsistent with status/error "
                f"rows={int(bad_failure_category_contract.sum())}"
            )

        schema_version, valid_schema_version = strict_integer_series(
            existing["result_schema_version"]
        )
        bad_schema_version = (
            ~valid_schema_version | schema_version.ne(int(RESULT_SCHEMA_VERSION))
        )
        if bool(bad_schema_version.any()):
            problems.append(
                "result_schema_version mismatch rows="
                f"{int(bad_schema_version.sum())}; expected "
                f"{int(RESULT_SCHEMA_VERSION)}"
            )
        seed_strategy_text = (
            existing["seed_strategy"].fillna("").astype(str).str.strip()
        )
        bad_seed_strategy = seed_strategy_text.ne(str(self.seed_strategy))
        if bool(bad_seed_strategy.any()):
            problems.append(
                "seed_strategy mismatch rows="
                f"{int(bad_seed_strategy.sum())}; expected {self.seed_strategy!r}"
            )
        seed_start, valid_seed_start = strict_integer_series(existing["seed_start"])
        bad_seed_start = ~valid_seed_start | seed_start.ne(int(self.seed_start))
        if bool(bad_seed_start.any()):
            problems.append(
                "seed_start mismatch rows="
                f"{int(bad_seed_start.sum())}; expected {int(self.seed_start)}"
            )
        if not PROTOCOL_DIGEST_COLUMNS.issubset(existing.columns):
            missing_protocol_cols = sorted(PROTOCOL_DIGEST_COLUMNS - set(existing.columns))
            problems.append(f"missing protocol digest columns={missing_protocol_cols}")
        else:
            bad_protocol_digest = invalid_protocol_digest_mask(existing)
            if bool(bad_protocol_digest.any()):
                problems.append(
                    "protocol_digest mismatch rows="
                    f"{int(bad_protocol_digest.sum())}"
                )

        key_cols = ["case_id", "method", "budget_mult", "repeat"]
        dup_mask = existing.duplicated(key_cols, keep=False)
        if bool(dup_mask.any()):
            dup_preview = (
                existing.loc[dup_mask, key_cols]
                .drop_duplicates()
                .head(10)
                .to_dict(orient="records")
            )
            problems.append(f"duplicate cells={dup_preview}")

        case_meta_by_id = self._resume_case_metadata()
        method_meta_by_name = self._resume_method_metadata()
        mismatches: list[dict[str, Any]] = []
        for _, row in existing.iterrows():
            case_id = str(row.get("case_id", ""))
            method_name = str(row.get("method", ""))
            budget_mult = _coerce_int(row.get("budget_mult"))
            repeat = _coerce_int(row.get("repeat"))
            if (
                case_id not in expected_cases
                or method_name not in expected_methods
                or budget_mult not in expected_budgets
                or repeat not in expected_repeats
            ):
                continue
            case_meta = case_meta_by_id.get(case_id, {})
            method_meta = method_meta_by_name.get(method_name, {})
            expected_values = {
                "case_source_path": case_meta.get("case_source_path"),
                "case_source_sha256": case_meta.get("case_source_sha256"),
                "domain": case_meta.get("domain"),
                "dim": case_meta.get("dim"),
                "n_output": case_meta.get("n_output"),
                "method_family": method_meta.get("method_family"),
                "method_source_path": method_meta.get("method_source_path"),
                "method_source_sha256": method_meta.get("method_source_sha256"),
                "n_train": (
                    int(budget_mult * int(case_meta["dim"]))
                    if budget_mult is not None and "dim" in case_meta
                    else None
                ),
                "n_test": self.n_test,
                "test_seed": self.test_seed,
                "seed": (
                    derive_seed(
                        case_id,
                        int(repeat),
                        strategy=self.seed_strategy,
                        seed_start=self.seed_start,
                    )
                    if repeat is not None
                    else None
                ),
            }
            bad_cols = [
                col
                for col, expected in expected_values.items()
                if expected is not None
                and not _same_resume_contract_value(col, row.get(col), expected)
            ]
            if bad_cols:
                mismatches.append({
                    "case_id": case_id,
                    "method": method_name,
                    "budget_mult": budget_mult,
                    "repeat": repeat,
                    "columns": bad_cols,
                })
        if mismatches:
            problems.append(f"metadata/protocol mismatches={mismatches[:10]}")

        if problems:
            raise ValueError(
                "--skip-existing cannot resume from a results.csv that is not "
                "a subset of the current declared grid. "
                f"File: {csv_path}. Problems: {'; '.join(problems)}. "
                "Use a clean output directory, filter the CSV explicitly, or "
                "rerun with the matching pool/budget/repeat configuration."
            )

    def _resume_case_metadata(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for case_id in self.case_ids:
            case = self.case_loader(case_id)
            meta = case.meta
            out[str(case_id)] = {
                **case_source_fingerprint(case),
                "domain": meta.domain,
                "dim": int(meta.dim_input),
                "n_output": int(meta.dim_output),
            }
        return out

    def _resume_method_metadata(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for method in self.methods:
            out[str(method.name)] = {
                "method_family": method.family,
                **method_source_fingerprint(method),
            }
        return out

    def _validate_resume_run_config(self, run_config_path: Path, csv_path: Path, *, total_cells: int) -> None:
        if run_config_path.is_symlink():
            raise ValueError(
                "--skip-existing cannot resume from a symlinked run_config.json. "
                f"File: {run_config_path}. Use a real startup snapshot in the "
                "output directory so the runner cannot trust an external target."
            )
        if not run_config_path.exists():
            raise ValueError(
                "--skip-existing requires a sibling run_config.json startup "
                "snapshot next to the existing results.csv. "
                f"Missing file: {run_config_path}. Restore the Stage-2 startup "
                "snapshot or resume from a clean output directory."
            )
        if not run_config_path.is_file() or run_config_path.stat().st_size == 0:
            raise ValueError(
                "--skip-existing cannot resume from an empty or non-file "
                f"run_config.json. File: {run_config_path}. Restore a valid "
                "Stage-2 startup snapshot before resuming."
            )
        try:
            payload = json.loads(run_config_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ValueError(
                "--skip-existing cannot resume from an unreadable or corrupt "
                f"run_config.json. File: {run_config_path}. Parser error: "
                f"{type(exc).__name__}: {exc}. Restore the startup snapshot or "
                "resume from a clean output directory."
            ) from exc
        if not isinstance(payload, dict):
            raise ValueError(
                "--skip-existing requires run_config.json to be a JSON object. "
                f"File: {run_config_path}. Restore the Stage-2 startup snapshot "
                "before resuming."
            )

        problems: list[str] = []
        if payload.get("artifact") != "doe-benchmark-stage2-run-config":
            problems.append(
                "artifact marker is not 'doe-benchmark-stage2-run-config'"
            )
        if _coerce_int(payload.get("result_schema_version")) != int(RESULT_SCHEMA_VERSION):
            problems.append(
                "result_schema_version does not match the current runner"
            )
        if _coerce_int(payload.get("total_cells")) != int(total_cells):
            problems.append("total_cells does not match the current declared grid")

        expected_output_dir = (
            _normalize_resume_output_dir_path(self.output_dir)
            if self.output_dir is not None
            else ""
        )
        observed_output_dir = _normalize_resume_output_dir_path(payload.get("output_dir"))
        if expected_output_dir and observed_output_dir and observed_output_dir != expected_output_dir:
            problems.append(
                "output_dir does not match the current resume directory"
            )

        case_metadata = self._resume_case_metadata()
        expected_case_rows = sorted(
            (
                str(case_id),
                str(meta["domain"]),
                int(meta["dim"]),
                int(meta["n_output"]),
                _normalize_resume_source_path(meta["case_source_path"]),
                str(meta["case_source_sha256"]),
            )
            for case_id, meta in case_metadata.items()
        )
        observed_case_rows: list[tuple[Any, ...]] = []
        cases_payload = payload.get("cases")
        if not isinstance(cases_payload, list) or not cases_payload:
            problems.append("cases entries are missing or malformed")
        else:
            for item in cases_payload:
                if not isinstance(item, dict):
                    observed_case_rows = []
                    problems.append("cases entries are missing or malformed")
                    break
                observed_case_rows.append(
                    (
                        str(item.get("case_id", "")).strip(),
                        str(item.get("domain", "")).strip(),
                        _coerce_int(item.get("dim")),
                        _coerce_int(item.get("n_output")),
                        _normalize_resume_source_path(item.get("case_source_path")),
                        str(item.get("case_source_sha256", "")).strip(),
                    )
                )
            if observed_case_rows and sorted(observed_case_rows) != expected_case_rows:
                problems.append("cases metadata/fingerprints drifted from the current runner")

        method_metadata = self._resume_method_metadata()
        expected_method_rows = sorted(
            (
                str(method.name),
                str(method.family),
                str(getattr(method, "parallel_backend", "loky") or "loky"),
                str(getattr(method, "memory_class", "light") or "light"),
                _normalize_resume_source_path(
                    method_metadata[str(method.name)]["method_source_path"]
                ),
                str(method_metadata[str(method.name)]["method_source_sha256"]),
            )
            for method in self.methods
        )
        observed_method_rows: list[tuple[Any, ...]] = []
        methods_payload = payload.get("methods")
        if not isinstance(methods_payload, list) or not methods_payload:
            problems.append("methods entries are missing or malformed")
        else:
            for item in methods_payload:
                if not isinstance(item, dict):
                    observed_method_rows = []
                    problems.append("methods entries are missing or malformed")
                    break
                observed_method_rows.append(
                    (
                        str(item.get("method", "")).strip(),
                        str(item.get("method_family", "")).strip(),
                        str(item.get("parallel_backend", "")).strip(),
                        str(item.get("memory_class", "")).strip(),
                        _normalize_resume_source_path(item.get("method_source_path")),
                        str(item.get("method_source_sha256", "")).strip(),
                    )
                )
            if observed_method_rows and sorted(observed_method_rows) != expected_method_rows:
                problems.append(
                    "methods metadata/fingerprints drifted from the current runner"
                )

        expected_budgets = [int(b) for b in self.budgets]
        observed_budgets = payload.get("budgets")
        if not isinstance(observed_budgets, list):
            problems.append("budgets are missing or malformed")
        else:
            parsed_budgets = [_coerce_int(item) for item in observed_budgets]
            if any(item is None for item in parsed_budgets) or parsed_budgets != expected_budgets:
                problems.append("budgets do not match the current declared grid")

        scalar_expectations = {
            "n_repeats": int(self.n_repeats),
            "n_test": int(self.n_test),
            "test_seed": int(self.test_seed),
            "seed_start": int(self.seed_start),
        }
        for key, expected in scalar_expectations.items():
            if _coerce_int(payload.get(key)) != expected:
                problems.append(f"{key} does not match the current runner")
        if str(payload.get("seed_strategy", "")).strip() != str(self.seed_strategy):
            problems.append("seed_strategy does not match the current runner")

        temp_storage = payload.get("temp_storage")
        if not isinstance(temp_storage, dict):
            problems.append("temp_storage startup snapshot is missing or malformed")
        else:
            if _coerce_int(temp_storage.get("schema_version")) != 1:
                problems.append("temp_storage schema_version is missing or invalid")
            if not str(temp_storage.get("resolved_path", "")).strip():
                problems.append("temp_storage resolved_path is missing")
            if not isinstance(temp_storage.get("exists"), bool):
                problems.append("temp_storage exists flag is missing or invalid")
            if not isinstance(temp_storage.get("writable"), bool):
                problems.append("temp_storage writable flag is missing or invalid")
            env = temp_storage.get("env")
            if not isinstance(env, dict):
                problems.append("temp_storage env snapshot is missing or malformed")
            else:
                for name in _TEMP_STORAGE_ENV_VARS:
                    if name not in env:
                        problems.append(f"temp_storage env snapshot missing {name}")

        torch_repro = payload.get("torch_reproducibility")
        if not isinstance(torch_repro, dict):
            problems.append("torch_reproducibility snapshot is missing or malformed")
        else:
            if _coerce_int(torch_repro.get("schema_version")) != 1:
                problems.append("torch_reproducibility schema_version is missing or invalid")
            if str(torch_repro.get("status", "")).strip() not in {"ok", "missing"}:
                problems.append("torch_reproducibility status is invalid")
            state = torch_repro.get("state")
            if str(torch_repro.get("status", "")).strip() == "ok":
                if not isinstance(state, dict):
                    problems.append("torch_reproducibility state is missing or malformed")
                else:
                    if state.get("deterministic_algorithms") is not True:
                        problems.append("torch deterministic_algorithms policy is not enabled")
                    if state.get("cudnn_benchmark") is True:
                        problems.append("torch cudnn benchmark is enabled")
                    if state.get("cuda_matmul_allow_tf32") is True:
                        problems.append("torch CUDA matmul TF32 is enabled")
                    if state.get("cudnn_allow_tf32") is True:
                        problems.append("torch cuDNN TF32 is enabled")

        if problems:
            raise ValueError(
                "--skip-existing cannot resume because the sibling "
                "run_config.json does not match the current declared Stage-2 "
                f"grid/protocol. results.csv={csv_path}; run_config.json={run_config_path}. "
                f"Problems: {'; '.join(problems)}. Restore the matching startup "
                "snapshot or resume from a clean output directory."
            )

    def _build_reference_set(self, case):
        """为 *case* 采样 + 评估参考测试集；返回 ``(X, Y)``。

        抽取为实例方法，以便作为可调用对象传给
        :func:`_call_with_soft_timeout`。

        保护措施
        --------
        * 如果 ``case.evaluate`` 返回全 NaN（损坏的 case），抛出
          :class:`RuntimeError`，让 runner 通过 ``ref_sets[case_id] = None``
          路径跳过该 case，而不是静默产生零行参考集。
        * 如果存活行太少（< 50），抛出 ``RuntimeError``，防止
          单元格指标计算在退化测试集上进行，导致下游统计无效。
        """
        X_test = case.sample(self.n_test, method="sobol", seed=self.test_seed)
        Y_test = case.evaluate(X_test)
        m_ok = ~np.any(np.isnan(Y_test), axis=1)
        n_dropped = int((~m_ok).sum())
        n_ok = int(m_ok.sum())
        if self.verbose and n_dropped:
            print(f"  [warn] dropped {n_dropped} NaN rows from reference")
        # 保护：拒绝退化的参考集。两个阈值：
        #   - 绝对：至少 1 行有效（否则每个指标都无定义）
        #   - 相对：至少 5% 的请求点存活（95% NaN 率表示损坏的 case，
        #     而非合法的边缘丢弃）
        # 相对测试在 n_test < 20 时跳过（冒烟/单元测试使用小参考集，
        # 5% 四舍五入为零）。
        if n_ok < 1:
            raise RuntimeError(
                f"reference-set has 0 valid rows out of {self.n_test} "
                f"requested ({n_dropped} NaN-dropped); case is broken or "
                f"sampling domain is entirely outside the valid region."
            )
        if self.n_test >= 20 and n_ok < max(1, int(0.05 * self.n_test)):
            raise RuntimeError(
                f"reference-set has only {n_ok} valid rows out of "
                f"{self.n_test} requested ({n_dropped} NaN-dropped); "
                f"that is below the 5% NaN-survival floor.  Likely a "
                f"broken case or out-of-range sampling domain."
            )
        return X_test[m_ok], Y_test[m_ok]

    @staticmethod
    def _stream_append_row(csv_path: Path, row: dict) -> None:
        """向 ``csv_path`` 追加一行，必要时演进 schema。

        行为：
          - 如果文件不存在，以行的 key 作为表头写入。
          - 如果文件存在且 ``row`` 只使用已有列，按现有列顺序追加
            （缺失的 key 填 NaN）。
          - 如果 ``row`` 引入了新列，用旧列 + 新列的并集重建文件
            （保留所有之前的行），然后追加。

        这保证流式 CSV 在运行中始终可被 ``pandas.read_csv`` 解析。
        """
        if csv_path.is_symlink():
            raise RuntimeError(
                f"refusing to append streamed results to symlinked results.csv: {csv_path}"
            )
        if not csv_path.exists():
            with open(csv_path, "w", encoding="utf-8", newline="") as fh:
                pd.DataFrame([row]).to_csv(fh, index=False)
                _flush_and_fsync(fh)
            _fsync_parent_dir(csv_path)
            return

        # 读取现有表头，不解析数据体
        with open(csv_path, encoding="utf-8", newline="") as fh:
            header_line = fh.readline().rstrip("\n").rstrip("\r")
        existing_cols = header_line.split(",") if header_line else []
        new_cols = [c for c in row.keys() if c not in existing_cols]

        if not new_cols:
            # 按现有列顺序追加，缺失的 key 填 NaN
            aligned = pd.DataFrame([row]).reindex(columns=existing_cols)
            with open(csv_path, "a", encoding="utf-8", newline="") as fh:
                aligned.to_csv(fh, index=False, header=False)
                _flush_and_fsync(fh)
            _fsync_parent_dir(csv_path)
            return

        # Schema 演进：用并集 schema 重写文件。先写同目录临时文件再
        # 原子替换，避免服务器进程在重写中途被杀时破坏已有结果。
        try:
            existing_df = pd.read_csv(csv_path)
        except Exception as exc:
            raise RuntimeError(
                f"cannot evolve streamed results schema because existing "
                f"CSV is not parseable: {csv_path} ({type(exc).__name__}: {exc})"
            ) from exc

        all_cols = list(existing_cols) + new_cols
        existing_df = existing_df.reindex(columns=all_cols)
        new_row_df = pd.DataFrame([row]).reindex(columns=all_cols)
        union_df = pd.concat([existing_df, new_row_df], ignore_index=True)
        tmp_path: Path | None = None
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="",
            dir=str(csv_path.parent),
            prefix=f".{csv_path.name}.tmp.",
            suffix=".csv",
            delete=False,
        ) as tmp:
            tmp_path = Path(tmp.name)
            union_df.to_csv(tmp, index=False)
            _flush_and_fsync(tmp)
        try:
            os.replace(tmp_path, csv_path)
            _fsync_parent_dir(csv_path)
        finally:
            if tmp_path is not None and tmp_path.exists():
                tmp_path.unlink()
