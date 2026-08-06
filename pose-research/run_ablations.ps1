$ErrorActionPreference = "Stop"
$python = "python"
$common = @(
  "--width", "32",
  "--graph-depth", "3",
  "--epochs", "8",
  "--batch-size", "32",
  "--train-limit", "24000",
  "--val-limit", "1800",
  "--workers", "4",
  "--seed", "3407"
)

& $python "train_kinepose.py" "--variant" "plain" "--run-name" "ablation-plain" @common
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $python "train_kinepose.py" "--variant" "no_graph" "--run-name" "ablation-no-graph" @common
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $python "train_kinepose.py" "--variant" "no_fields" "--run-name" "ablation-no-fields" @common
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $python "train_kinepose.py" "--variant" "full" "--run-name" "ablation-full" @common
exit $LASTEXITCODE
