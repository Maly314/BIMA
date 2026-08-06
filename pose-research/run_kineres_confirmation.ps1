$ErrorActionPreference = "Stop"
$common = @(
  "--decoder-channels", "96",
  "--graph-depth", "3",
  "--epochs", "8",
  "--batch-size", "32",
  "--train-limit", "24000",
  "--val-limit", "1800",
  "--workers", "4",
  "--seed", "9019"
)
& python "train_kineres.py" "--variant" "plain" "--run-name" "confirm-kineres-plain-seed9019" @common
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& python "train_kineres.py" "--variant" "graph" "--run-name" "confirm-kineres-graph-seed9019" @common
exit $LASTEXITCODE
