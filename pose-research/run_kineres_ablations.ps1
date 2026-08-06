param([int]$WaitForPid = 0)
$ErrorActionPreference = "Stop"
if ($WaitForPid -gt 0 -and (Get-Process -Id $WaitForPid -ErrorAction SilentlyContinue)) {
  Wait-Process -Id $WaitForPid
}
$common = @(
  "--decoder-channels", "96",
  "--graph-depth", "3",
  "--epochs", "8",
  "--batch-size", "32",
  "--train-limit", "24000",
  "--val-limit", "1800",
  "--workers", "4",
  "--seed", "3407"
)
& python "train_kineres.py" "--variant" "plain" "--run-name" "kineres-plain" @common
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& python "train_kineres.py" "--variant" "graph" "--run-name" "kineres-graph" @common
exit $LASTEXITCODE
