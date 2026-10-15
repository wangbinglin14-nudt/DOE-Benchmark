#!/usr/bin/env python
"""将指定 (case, method, budget, repeat) 单元标记为 timeout 并追加到 results.csv。

用途
----
本项目的 cell 软超时基于 ``ThreadPoolExecutor``（见 ``benchmark/core.py``
的 ``_run_cell``），而 BoTorch / GPyTorch 方法声明 ``parallel_backend=
'threading'``——它们的 native MCMC / GP 求解器一旦在 C/Fortran 层卡死并霸占
GIL，基于线程的超时回调无法执行，cell 会无限期挂起。本脚本用于：

1. 手动恢复：把已确认挂死的 (case, method) 组合标记为 timeout；
2. 配合外部 SIGKILL 看门狗自动恢复（见 ``tools/sweep_watchdog.sh``）。

正确性保证
----------
- ``seed`` 用 ``benchmark.core.derive_seed`` 复算，与 runner / audit 完全一致；
- ``protocol_digest`` 用 ``protocol_digest_for_row`` 基于「行自身字段」自洽生成
  （audit 会重算并比对，因此自洽即合法）；
- case / method 静态协议字段从已存在的真实行拷贝，sha256 等完全一致。

幂等：仅追加 CSV 中尚不存在的 (case, method, budget, repeat) 单元。
"""
from __future__ import annotations

import argparse
import os
import socket
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from benchmark.core import derive_seed
from benchmark.runtime_contracts import PROTOCOL_DIGEST_FIELDS, protocol_digest_for_row

# case 级协议字段（来自 case 模板行）
_CASE_FIELDS = (
    "case_id",
    "case_source_path",
    "case_source_sha256",
    "domain",
    "dim",
    "n_output",
)
# method 级协议字段（来自 method 模板行）
_METHOD_FIELDS = (
    "method",
    "method_family",
    "method_source_path",
    "method_source_sha256",
)
# seed / 协议盐字段（来自任一模板行，同一次 run 内全局一致）
_RUN_FIELDS = ("seed_strategy", "seed_start", "test_seed", "result_schema_version")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _utc_iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds")


def _first_row(df: pd.DataFrame, mask) -> pd.Series | None:
    sub = df[mask]
    return sub.iloc[0] if len(sub) else None


def _case_template_from_testbed(case_id: str) -> pd.Series:
    """CSV 中尚无该 case 的行时，从 doe_testbed 实加载构造 case 模板。

    复用 runner 的 case_source_fingerprint，保证 case_source_path/sha256 与
    _validate_resume_grid 的 _resume_case_metadata 完全一致。
    """
    import doe_testbed as db
    from benchmark.core import case_source_fingerprint

    case = db.get_case(case_id)
    meta = db.get_case_meta(case_id)
    fp = case_source_fingerprint(case)
    return pd.Series({
        "case_id": case_id,
        "case_source_path": fp["case_source_path"],
        "case_source_sha256": fp["case_source_sha256"],
        "domain": meta.domain,
        "dim": int(meta.dim_input),
        "n_output": int(meta.dim_output),
    })


def build_timeout_row(
    case_tmpl: pd.Series,
    method_tmpl: pd.Series,
    columns: list[str],
    *,
    budget_mult: int,
    repeat: int,
    n_test: int,
    timeout_s: float,
    reason: str,
) -> dict:
    """构造一行 audit-valid 的 timeout 记录。"""
    dim = int(case_tmpl["dim"])
    case_id = str(case_tmpl["case_id"])
    method = str(method_tmpl["method"])
    n_train = int(budget_mult) * dim

    row: dict = {col: "" for col in columns}
    # case 级
    for f in _CASE_FIELDS:
        row[f] = case_tmpl[f]
    # method 级
    for f in _METHOD_FIELDS:
        row[f] = method_tmpl[f]
    # run 级（seed_strategy / seed_start / test_seed / schema 版本）
    for f in _RUN_FIELDS:
        row[f] = case_tmpl[f] if f in case_tmpl else method_tmpl[f]

    # cell 级动态字段
    row["budget_mult"] = int(budget_mult)
    row["n_train"] = n_train
    row["n_test"] = int(n_test)
    row["repeat"] = int(repeat)
    row["seed"] = int(
        derive_seed(
            case_id,
            repeat,
            strategy=str(row["seed_strategy"]),
            seed_start=int(row["seed_start"]),
        )
    )
    row["worker_host"] = socket.gethostname()
    row["worker_pid"] = int(os.getpid())
    started = datetime.now(timezone.utc)
    # 关键：finished 必须与 t_total_s 一致，否则 _validate_resume_grid 的
    # invalid_runtime_duration_mask 会以「时间戳与运行时长不符」拒绝整份 CSV。
    finished = started + timedelta(seconds=float(timeout_s))
    row["cell_started_utc"] = _utc_iso(started)
    row["cell_finished_utc"] = _utc_iso(finished)
    # rss_* 不在 PROTOCOL_DIGEST_FIELDS 中，改它们不影响 digest；取一致正值，
    # 确保通过 rss_gb_peak >= rss_gb_start/end 校验。
    row["rss_gb_start"] = 0.0
    row["rss_gb_end"] = 0.0
    row["rss_gb_peak"] = 0.0
    row["status"] = "timeout"
    row["failure_category"] = "timeout"
    row["error"] = f"timeout after {timeout_s:.0f}s ({method} on {case_id}, n_train={n_train}) [{reason}]"
    row["t_total_s"] = float(timeout_s)

    # 自洽协议摘要（必须最后算，依赖上面的协议字段）
    row["protocol_digest"] = protocol_digest_for_row(row)
    return row


def find_stuck_target(df: pd.DataFrame, expected_per_case: int = 2400):
    """自动定位「当前卡死」的 (case, method)。

    策略：找最后一个进行中的 case（有行但未满 expected_per_case），在其中找
    BoTorch 方法里 cell 数最少的（最可能是挂死的那一个）。BoTorch 方法是唯一
    会因 native 代码卡死、绕过线程软超时的方法，因此优先标记它们。
    返回 (case_id, method) 或 None。
    """
    case_counts = df.groupby("case_id").size()
    in_progress = case_counts[(case_counts > 0) & (case_counts < expected_per_case)]
    if in_progress.empty:
        return None
    # 最新进行中的 case = 行数最大的那个（runner 按顺序处理）
    case_id = in_progress.idxmax()
    sub = df[df["case_id"] == case_id]
    method_counts = sub.groupby("method").size()
    incomplete = method_counts[method_counts < (expected_per_case // 20)]  # 每 method 应 120
    # 优先 BoTorch 方法
    botorch = [m for m in incomplete.index if m in ("botorch_qei", "botorch_saasbo")]
    candidates = botorch if botorch else list(incomplete.index)
    if not candidates:
        return None
    # cell 数最少的（最早开始卡的那一个）
    target = min(candidates, key=lambda m: method_counts[m])
    return case_id, target


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", type=Path, help="results.csv 路径")
    ap.add_argument("--case", help="case_id（与 --method 一起用于手动指定；省略则 --auto 自动检测）")
    ap.add_argument("--method", help="method name")
    ap.add_argument(
        "--auto", action="store_true",
        help="自动检测当前进行中 case 里最可能卡死的 BoTorch 方法并标记",
    )
    ap.add_argument(
        "--budgets", type=int, nargs="+", default=[5, 10, 20, 50],
        help="要标记的 budget_mult 列表（默认 5 10 20 50）",
    )
    ap.add_argument("--n-repeats", type=int, default=30, help="每个 cell 的重复数（默认 30）")
    ap.add_argument("--n-test", type=int, default=1000, help="测试集大小（默认 1000）")
    ap.add_argument("--timeout-s", type=float, default=1800.0, help="记录的 t_total_s（默认 1800=BoTorch 超时）")
    ap.add_argument("--reason", default="native-code hang", help="标记原因，写入 error 字段")
    args = ap.parse_args()

    df = pd.read_csv(args.csv, low_memory=False)
    columns = list(df.columns)

    if args.auto:
        target = find_stuck_target(df)
        if target is None:
            print("[auto] 没有进行中的 case，无需标记。")
            return 0
        args.case, args.method = target
        print(f"[auto] 检测到卡死目标: case={args.case} method={args.method}")
    elif not args.case or not args.method:
        print("[error] 必须提供 --auto，或同时提供 --case 和 --method。", file=sys.stderr)
        return 2

    case_tmpl = _first_row(df, df["case_id"] == args.case)
    if case_tmpl is None:
        # CSV 中尚无该 case（未启动），从 doe_testbed 实加载
        try:
            case_tmpl = _case_template_from_testbed(args.case)
            # 补 run 级字段（seed_strategy 等取自任一已有行）
            any_row = df.iloc[0]
            for f in _RUN_FIELDS:
                case_tmpl[f] = any_row[f]
            print(f"[info] case {args.case!r} 在 CSV 中无记录，已从 doe_testbed 加载 case 元数据。")
        except Exception as exc:
            print(f"[error] CSV 中找不到 case_id={args.case!r} 且 doe_testbed 加载失败: {exc}", file=sys.stderr)
            return 2
    method_tmpl = _first_row(df, df["method"] == args.method)
    if method_tmpl is None:
        print(f"[error] CSV 中找不到 method={args.method!r} 的任何行，无法获取 method 协议字段。", file=sys.stderr)
        return 2

    # 已存在的 (budget, repeat)
    existing = df[(df["case_id"] == args.case) & (df["method"] == args.method)]
    have = set(zip(existing["budget_mult"].astype(int), existing["repeat"].astype(int)))

    new_rows: list[dict] = []
    for b in args.budgets:
        for r in range(args.n_repeats):
            if (b, r) in have:
                continue
            new_rows.append(
                build_timeout_row(
                    case_tmpl, method_tmpl, columns,
                    budget_mult=b, repeat=r, n_test=args.n_test,
                    timeout_s=args.timeout_s, reason=args.reason,
                )
            )

    if not new_rows:
        print(f"[ok] {args.case} × {args.method}: 无缺失 cell，无需标记。")
        return 0

    # 追加（不重写整个文件，避免破坏其它行；用 csv 模式追加）
    import csv
    write_header = not args.csv.exists() or args.csv.stat().st_size == 0
    with open(args.csv, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        if write_header:
            w.writeheader()
        for row in new_rows:
            w.writerow(row)

    print(f"[ok] {args.case} × {args.method}: 追加 {len(new_rows)} 行 timeout "
          f"(budgets={args.budgets}, repeats=0..{args.n_repeats - 1})")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
