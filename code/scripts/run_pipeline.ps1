<#
.SYNOPSIS
    One-click DOE-Benchmark pipeline launcher for Windows (PowerShell 7+).

.DESCRIPTION
    Thin wrapper around `scripts/run_pipeline.py`.  Responsibilities:

      * Locate a usable Python interpreter (prefer the local venv).
      * Pin BLAS / OpenMP thread caps so joblib parallelism is the only
        layer of parallelism — without this MKL spawns 16-256 threads
        per worker and the kernel scheduler thrashes on a 16-core box.
      * Forward every remaining argument to `run_pipeline.py` so the
        full Python CLI surface is reachable from PowerShell without
        argument re-parsing here.

    All real logic lives in the Python orchestrator; this wrapper is
    intentionally minimal so behaviour stays identical between Windows
    and Linux launchers (`scripts/run_pipeline.sh`).

.PARAMETER Profile
    'smoke' (~5 min sanity check, default) or 'full' for no-sweep dry-run /
    Stage-1 preflight checks.  Windows workstations must not launch the
    publication full sweep.

.PARAMETER ExtraArgs
    Any flags to pass through to `run_pipeline.py` (e.g.
    less common flags not exposed directly by this wrapper).

.PARAMETER DryRun
    Print the resolved commands without executing subprocesses.

.PARAMETER ToStage
    Forward `--to-stage N`; on Windows, full-profile non-dry-run commands are
    allowed only when `ToStage` is 1.

.PARAMETER FromStage
    Forward `--from-stage N`; full-profile non-dry-run commands remain blocked
    on Windows.

.PARAMETER Only
    Forward one or more `--only` stage names.

.PARAMETER Skip
    Forward one or more `--skip` stage names.

.EXAMPLE
    pwsh -File scripts/run_pipeline.ps1 -Profile smoke
        # Smoke run: 1 case × 1 method × 2 budgets × 2 repeats.

.EXAMPLE
    pwsh -File scripts/run_pipeline.ps1
        # Smoke run: 1 case x default-pool mini sweep.

.EXAMPLE
    pwsh -File scripts/run_pipeline.ps1 -Profile full -ToStage 1
        # Full-profile Stage-1 preflight only; does not enter Stage 2.

.EXAMPLE
    pwsh -File scripts/run_pipeline.ps1 -Profile full -DryRun
        # Print the resolved subprocess commands without executing.
#>
[CmdletBinding(PositionalBinding = $false)]
param(
    [ValidateSet('smoke', 'full')]
    [string]$Profile = 'smoke',

    [switch]$DryRun,

    [int]$ToStage = 0,

    [int]$FromStage = 0,

    [string[]]$Only = @(),

    [string[]]$Skip = @(),

    # Catches every other token so the operator can spell out the
    # remaining run_pipeline.py CLI without re-declaring it here.
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs = @()
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot  = Split-Path -Parent $scriptDir

# ── Locate Python ────────────────────────────────────────────────────────
# Order: $env:PYTHON, repo-local .venv, py launcher, python on PATH.
function Resolve-Python {
    if ($env:PYTHON -and (Test-Path $env:PYTHON)) { return $env:PYTHON }
    $venvPython = Join-Path $repoRoot '.venv\Scripts\python.exe'
    if (Test-Path $venvPython) { return $venvPython }
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    $pyLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($pyLauncher) { return $pyLauncher.Source }
    throw "Cannot find a Python interpreter.  Install Python 3.10+ or set `\$env:PYTHON."
}
$python = Resolve-Python

# ── Pin BLAS threads BEFORE launching Python ────────────────────────────
# These must be set on the *parent* process: child processes inherit the
# environment block, but if we set them inside Python it's already too
# late for libraries that are imported before the env-var read.
foreach ($var in @(
    'OMP_NUM_THREADS',
    'OPENBLAS_NUM_THREADS',
    'MKL_NUM_THREADS',
    'NUMEXPR_NUM_THREADS',
    'BLIS_NUM_THREADS'
)) {
    if (-not (Get-Item -Path "Env:$var" -ErrorAction SilentlyContinue)) {
        Set-Item -Path "Env:$var" -Value '1'
    }
}
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUNBUFFERED = '1'

# ── Banner so the operator can confirm before the long run ──────────────
Write-Host ''
Write-Host ('=' * 78)
Write-Host ' DOE-Benchmark one-click launcher (Windows / PowerShell)'
Write-Host ('=' * 78)
Write-Host ('  repo root  : {0}' -f $repoRoot)
Write-Host ('  python     : {0}' -f $python)
Write-Host ('  profile    : {0}' -f $Profile)
Write-Host ('  dry-run    : {0}' -f $DryRun.IsPresent)
Write-Host ('  to-stage   : {0}' -f ($(if ($ToStage -gt 0) { $ToStage } else { '<none>' })))
Write-Host ('  extra args : {0}' -f ($(if ($ExtraArgs) { $ExtraArgs -join ' ' } else { '<none>' })))
Write-Host ('=' * 78)
Write-Host ''

$forwardArgs = @()
if ($DryRun) { $forwardArgs += '--dry-run' }
if ($ToStage -gt 0) { $forwardArgs += @('--to-stage', "$ToStage") }
if ($FromStage -gt 0) { $forwardArgs += @('--from-stage', "$FromStage") }
foreach ($stage in $Only) { $forwardArgs += @('--only', $stage) }
foreach ($stage in $Skip) { $forwardArgs += @('--skip', $stage) }
$forwardArgs += $ExtraArgs

function Test-HasFlag {
    param([string[]]$Tokens, [string]$Flag)
    foreach ($arg in $Tokens) {
        if ($arg -eq $Flag -or $arg.StartsWith("$Flag=")) { return $true }
    }
    return $false
}

function Get-FlagValue {
    param([string[]]$Tokens, [string]$Flag)
    for ($i = 0; $i -lt $Tokens.Count; $i++) {
        $arg = $Tokens[$i]
        if ($arg.StartsWith("$Flag=")) {
            return $arg.Substring($Flag.Length + 1)
        }
        if ($arg -eq $Flag -and ($i + 1) -lt $Tokens.Count) {
            return $Tokens[$i + 1]
        }
    }
    return $null
}

function Test-IsStage1Only {
    param([string[]]$Tokens)
    $toStage = Get-FlagValue -Tokens $Tokens -Flag '--to-stage'
    return ($toStage -eq '1')
}

if ($Profile -eq 'full') {
    $isDryRun = Test-HasFlag -Tokens $forwardArgs -Flag '--dry-run'
    $isStage1Only = Test-IsStage1Only -Tokens $forwardArgs
    if (-not $isDryRun -and -not $isStage1Only) {
        Write-Host '[error] Windows run_pipeline.ps1 is smoke/preflight only.' -ForegroundColor Red
        Write-Host '        Do not launch the 34 x 20 x 4 x 30 Nature CS full sweep locally.'
        Write-Host '        Allowed Windows examples:'
        Write-Host '          pwsh -File scripts\run_pipeline.ps1'
        Write-Host '          pwsh -File scripts\run_pipeline.ps1 -Profile full -ToStage 1'
        Write-Host '          pwsh -File scripts\run_pipeline.ps1 -Profile full -DryRun'
        Write-Host '        Formal Linux server launch:'
        Write-Host '          python scripts/check_server_launch_ready.py'
        Write-Host '          bash scripts/run_pipeline.sh'
        exit 2
    }
}

# ── Forward to the Python orchestrator ──────────────────────────────────
$cmdArgs = @(
    (Join-Path $scriptDir 'run_pipeline.py'),
    '--profile', $Profile
) + $forwardArgs

if ($Profile -eq 'full' -and ($forwardArgs -notcontains '--confirm-full-run')) {
    $cmdArgs = @((Join-Path $scriptDir 'run_pipeline.py'), '--confirm-full-run', '--profile', $Profile) + $forwardArgs
}

# Use Push-Location so failures still restore the original PWD.
Push-Location $repoRoot
try {
    & $python @cmdArgs
    $code = $LASTEXITCODE
} finally {
    Pop-Location
}

if ($code -ne 0) {
    Write-Host ''
    Write-Host ('[error] run_pipeline.py exited with code {0}' -f $code) -ForegroundColor Red
}
exit $code
