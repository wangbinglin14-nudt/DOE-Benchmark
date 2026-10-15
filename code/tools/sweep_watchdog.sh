#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────
# sweep_watchdog.sh — 全量 sweep 的自愈看门狗
#
# 问题：BoTorch / GPyTorch 方法声明 parallel_backend='threading'，其 native
# MCMC / GP 求解器在 C/Fortran 层卡死并霸占 GIL 时，基于 ThreadPoolExecutor
# 的 cell 软超时（benchmark/core.py: _run_cell）无法触发，cell 无限挂起。
#
# 本脚本在外部监控 results.csv 的行数增长：
#   - 停滞超过 STALL_SECONDS（默认 40 min > bo-timeout 30 min）→ 判定挂死
#   - SIGKILL 整个 sweep 进程树（SIGKILL 不受 GIL / native 代码影响）
#   - 调用 tools/mark_cells_timeout.py --auto 把卡死的 BoTorch cell 标成 timeout
#   - 自动重启 sweep（--skip-existing 跳过已完成 cell）
#   - sweep 正常退出 (rc=0) → 完成
#
# 产出的 timeout 行复用 runner 的 derive_seed + protocol_digest_for_row，
# audit 合法（见 mark_cells_timeout.py）。
# ──────────────────────────────────────────────────────────────────────────
set -u

PYTHON=/home/user/.conda/envs/wbl/bin/python
ROOT=/home/user/Desktop/DOE
CSV="$ROOT/benchmark/results/full_benchmark/results.csv"
WATCH_LOG="$ROOT/logs/watchdog.log"
STALL_SECONDS=2400        # 40 min 零增长 = 挂死（bo-timeout=1800s=30min）
POLL=180                  # 每 3 min 检查一次行数
MAX_RESTARTS=80

CASES="tuned_mass_damper_3d airfoil_stall_2d fin_heat_sink_4d multilayer_wall_4d pressure_vessel_4d rocket_isp_4d waverider_4d airfoil_transonic_5d cantilever_beam_5d casting_solidification_5d hetero_conduction_2d laminar_flame_speed_5d membrane_reactor_5d plate_with_hole_5d soil_consolidation_5d supersonic_intake_5d supersonic_nozzle_5d thin_plate_buckling_5d variable_beam_5d dam_seepage_6d fixed_bed_reactor_6d groundwater_contamination_6d rotor_bemt_6d solar_pv_yield_6d wind_turbine_power_6d wing_vlm_6d battery_thermal_7d aircraft_conceptual_8d multilayer_wall_8d oil_reservoir_9d ten_bar_truss_10d variable_beam_10d multilayer_wall_16d variable_beam_20d"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$WATCH_LOG"; }

kill_sweep() {
  local pid=$1
  log "SIGKILL sweep 主进程 pid=$pid 及其 worker 树..."
  kill -9 "$pid" 2>/dev/null
  sleep 2
  # 兜底：杀掉所有残留的 benchmark worker（watchdog 自身是 bash，不会被匹配）
  pkill -9 -f run_full_benchmark 2>/dev/null
  sleep 3
}

restart=0
while [ $restart -lt $MAX_RESTARTS ]; do
  restart=$((restart + 1))
  log "==================== 第 $restart 轮启动 sweep ===================="

  cd "$ROOT" || exit 1
  # 启动前清理 RSS 测量竞态坏行，否则 _validate_resume_grid 会 rc=1 拒绝整份 CSV
  PYTHONPATH="$ROOT" "$PYTHON" "$ROOT/tools/sanitize_results_rss.py" "$CSV" 2>&1 | tee -a "$WATCH_LOG"
  PYTHONPATH="$ROOT" "$PYTHON" -u benchmark/scripts/run_full_benchmark.py \
    --cases $CASES \
    --pool nature --budgets 5 10 20 50 \
    --n-repeats 30 --n-test 1000 \
    --seed-strategy case_repeat --seed-start 1000 \
    --n-jobs 16 --bo-n-jobs 4 --mem-cap-gb 56.0 \
    --timeout-per-cell 600.0 --bo-timeout-per-cell 1800.0 \
    --ref-timeout-s 600.0 --heartbeat-s 60.0 \
    --output-dir benchmark/results/full_benchmark \
    --skip-existing --confirm-large-run \
    --require-botorch --require-formal-case-backends --require-runtime-monitoring \
    >> "$ROOT/logs/pipeline_full.log" 2>&1 &
  SWEEP_PID=$!
  log "sweep 启动 pid=$SWEEP_PID"

  last_rows=0
  stall_since=0
  killed=0
  while kill -0 "$SWEEP_PID" 2>/dev/null; do
    sleep "$POLL"
    rows=$(wc -l < "$CSV" 2>/dev/null || echo 0)
    rows=$((rows - 1))   # 去掉表头
    if [ "$rows" -le "$last_rows" ]; then
      stall_since=$((stall_since + POLL))
      log "停滞: $rows 行 (零增长 ${stall_since}s / 阈值 ${STALL_SECONDS}s)"
      if [ "$stall_since" -ge "$STALL_SECONDS" ]; then
        log "⚠️  判定挂死，执行恢复"
        kill_sweep "$SWEEP_PID"
        killed=1
        log "标记卡死 cell 为 timeout..."
        PYTHONPATH="$ROOT" "$PYTHON" "$ROOT/tools/mark_cells_timeout.py" "$CSV" --auto 2>&1 | tee -a "$WATCH_LOG"
        break
      fi
    else
      if [ "$stall_since" -ne 0 ]; then log "恢复增长: $last_rows → $rows 行"; fi
      stall_since=0
      last_rows=$rows
    fi
  done

  if [ "$killed" -eq 1 ]; then
    log "恢复完成，进入下一轮重启"
    continue
  fi

  # sweep 自行退出
  wait "$SWEEP_PID"; rc=$?
  if [ "$rc" -eq 0 ]; then
    log "✅✅✅ sweep 正常完成 (rc=0)！全部 cell 已就绪。"
    exit 0
  else
    log "sweep 异常退出 rc=$rc，重启..."
  fi
done

log "❌ 达到最大重启次数 $MAX_RESTARTS，放弃。请人工排查。"
exit 1
