<#
.SYNOPSIS
    Guarded legacy local launcher for a 16-GB / 8-core Windows host.

.DESCRIPTION
    This script no longer launches the pre-registered full sweep from a local
    Windows workstation.  The Nature CS headline archive must be produced on
    the designated Linux server through scripts/run_pipeline.sh.  Keep this
    file only as a frozen protocol/case-list diagnostic for reviewers and CI.
#>
[CmdletBinding()]
param(
    [int]$WallClockH = 72,
    [int]$NJobs      = 8,
    [int]$BoNJobs    = 1,
    [double]$MemCapGb = 12.0
)

$ErrorActionPreference = "Stop"

$RepoRoot   = Split-Path -Parent $PSScriptRoot
$Python     = if ($env:PYTHON) { $env:PYTHON } elseif (Test-Path "C:\Miniconda3\python.exe") { "C:\Miniconda3\python.exe" } else { "python" }
$RunScript  = Join-Path $RepoRoot "benchmark\scripts\run_full_benchmark.py"
$OutDir     = Join-Path $RepoRoot "benchmark\results\full_benchmark"
$BatWrapper = Join-Path $OutDir "_local_launch.bat"
$LogOut     = Join-Path $OutDir "run.log"
$LogErr     = Join-Path $OutDir "run.err"

$ValidatorScript = Join-Path $RepoRoot "scripts\validate_run_config.py"
$DependencyScript = Join-Path $RepoRoot "scripts\check_optional_dependencies.py"
$Cases = @(
    "tuned_mass_damper_3d",
    "airfoil_stall_2d", "fin_heat_sink_4d", "multilayer_wall_4d",
    "pressure_vessel_4d", "rocket_isp_4d", "waverider_4d",
    "airfoil_transonic_5d", "cantilever_beam_5d", "casting_solidification_5d",
    "hetero_conduction_2d", "laminar_flame_speed_5d", "membrane_reactor_5d",
    "plate_with_hole_5d", "soil_consolidation_5d", "supersonic_intake_5d",
    "supersonic_nozzle_5d", "thin_plate_buckling_5d", "variable_beam_5d",
    "dam_seepage_6d", "fixed_bed_reactor_6d", "groundwater_contamination_6d",
    "rotor_bemt_6d", "solar_pv_yield_6d", "wind_turbine_power_6d", "wing_vlm_6d",
    "battery_thermal_7d",
    "aircraft_conceptual_8d", "multilayer_wall_8d",
    "oil_reservoir_9d",
    "ten_bar_truss_10d", "variable_beam_10d",
    "multilayer_wall_16d", "variable_beam_20d"
)

Write-Host ""
Write-Host "================================================================" -ForegroundColor Yellow
Write-Host " Local full-sweep launch is disabled" -ForegroundColor Yellow
Write-Host "================================================================" -ForegroundColor Yellow
Write-Host "This Windows helper is retained only as a frozen protocol diagnostic."
Write-Host "Do not launch the 34 x 20 x 4 x 30 Nature CS sweep from the local workstation."
Write-Host ""
Write-Host "Allowed local checks:"
Write-Host "  pwsh -File scripts\run_pipeline.ps1 -Profile smoke"
Write-Host "  python scripts\check_server_launch_ready.py"
Write-Host "  python scripts\run_case_physics_tests.py -q"
Write-Host ""
Write-Host "Formal Linux server launch:"
Write-Host "  python scripts/check_server_launch_ready.py"
Write-Host "  bash scripts/run_pipeline.sh"
Write-Host ""
Write-Host "Frozen case-list diagnostic: $($Cases.Count) cases are listed in this script."
exit 2

$PythonCommand = Get-Command $Python -ErrorAction SilentlyContinue
if (-not $PythonCommand -and -not (Test-Path $Python)) { throw "Python not found: $Python" }
if (-not (Test-Path $RunScript)) { throw "Run script not found: $RunScript" }
if (-not (Test-Path $OutDir)) { New-Item -ItemType Directory -Path $OutDir | Out-Null }

$ValidatorArgs = @(
    "$ValidatorScript",
    "--pool", "nature",
    "--budgets", "5", "10", "20", "50",
    "--n-repeats", "30",
    "--n-test", "1000",
    "--seed-strategy", "case_repeat",
    "--seed-start", "1000",
    "--cases"
)
$ValidatorArgs += $Cases
if ($env:ALLOW_DEVIATION -eq "1") {
    $ValidatorArgs += @("--allow-deviation", "--output-dir", "$OutDir")
}
& $Python @ValidatorArgs
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "ERROR: pre-flight validation failed. Set `$env:ALLOW_DEVIATION = '1' to override." -ForegroundColor Red
    exit 1
}
Write-Host ""

$DependencyArgs = @(
    "$DependencyScript",
    "--pool", "nature",
    "--require-botorch",
    "--expected-methods", "20",
    "--require-formal-case-backends",
    "--require-runtime-monitoring"
)
& $Python @DependencyArgs
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "ERROR: optional dependency pre-flight failed. Fix BoTorch/GPyTorch/Torch, formal case backends, and psutil runtime monitoring before the headline sweep." -ForegroundColor Red
    exit 1
}
Write-Host ""

$CaseArgs = $Cases -join " "

$PyArgs = @(
    "`"$RunScript`""
    "--cases $CaseArgs"
    "--pool nature"
    "--budgets 5 10 20 50"
    "--n-repeats 30"
    "--n-test 1000"
    "--seed-strategy case_repeat"
    "--seed-start 1000"
    "--n-jobs $NJobs"
    "--bo-n-jobs $BoNJobs"
    "--mem-cap-gb $MemCapGb"
    "--skip-existing"
    "--timeout-per-cell 600"
    "--bo-timeout-per-cell 1800"
    "--ref-timeout-s 600"
    "--heartbeat-s 60"
    "--max-wall-clock-h $WallClockH"
    "--output-dir `"$OutDir`""
    "--confirm-large-run"
    "--require-botorch"
    "--require-formal-case-backends"
    "--require-runtime-monitoring"
) -join " "

$BatBody = @"
@echo off
set OMP_NUM_THREADS=1
set OPENBLAS_NUM_THREADS=1
set MKL_NUM_THREADS=1
set NUMEXPR_NUM_THREADS=1
set BLIS_NUM_THREADS=1
"$Python" $PyArgs 1>>"$LogOut" 2>>"$LogErr"
"@
Set-Content -Path $BatWrapper -Value $BatBody -Encoding ASCII

$process = Start-Process `
    -FilePath "cmd.exe" `
    -ArgumentList @("/c", "`"$BatWrapper`"") `
    -WorkingDirectory $RepoRoot `
    -WindowStyle Hidden `
    -PassThru

if ($null -eq $process -or $process.Id -le 0) {
    throw "Start-Process failed to launch $BatWrapper"
}

$pid_ = $process.Id
Write-Host ""
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host " Unified single-phase sweep launched (Windows 16 GB local)" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "  PID            : $pid_"
Write-Host "  Cases          : $($Cases.Count)"
Write-Host "  Pool           : nature"
Write-Host "  Budgets        : 5 10 20 50 (* d)"
Write-Host "  Repeats        : 30"
Write-Host "  Seed strategy  : case_repeat, seed_start=1000"
Write-Host "  n_jobs (light) : $NJobs"
Write-Host "  n_jobs (heavy) : $BoNJobs"
Write-Host "  mem_cap_gb     : $MemCapGb"
Write-Host "  Wall-clock cap : $WallClockH h"
Write-Host "  Output         : $OutDir"
Write-Host ""
Write-Host "Monitor:"
Write-Host "  Get-Content `"$LogOut`" -Wait -Tail 30"
Write-Host "  python scripts\_check_sweep_progress.py"
Write-Host ""
Write-Host "Stop cleanly:"
Write-Host "  Stop-Process -Id $pid_"
Write-Host ""
