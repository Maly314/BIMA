param([int]$WaitForPid = 0)
$ErrorActionPreference = "Stop"
if ($WaitForPid -gt 0 -and (Get-Process -Id $WaitForPid -ErrorAction SilentlyContinue)) {
  Wait-Process -Id $WaitForPid
}
$common = @(
  "--width", "32",
  "--graph-depth", "3",
  "--epochs", "8",
  "--batch-size", "32",
  "--train-limit", "24000",
  "--val-limit", "1800",
  "--workers", "4",
  "--seed", "9019"
)
& python "train_kinepose.py" "--variant" "plain" "--run-name" "confirm-plain-seed9019" @common
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& python "train_kinepose.py" "--variant" "hybrid_fast" "--run-name" "confirm-fast-seed9019" @common
exit $LASTEXITCODE
