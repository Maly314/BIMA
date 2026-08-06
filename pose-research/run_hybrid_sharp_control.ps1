param([int]$WaitForPid = 0)
$ErrorActionPreference = "Stop"
if ($WaitForPid -gt 0 -and (Get-Process -Id $WaitForPid -ErrorAction SilentlyContinue)) {
  Wait-Process -Id $WaitForPid
}
& python "train_kinepose.py" `
  "--variant" "hybrid_sharp" `
  "--run-name" "ablation-hybrid-sharp" `
  "--width" "32" `
  "--graph-depth" "3" `
  "--epochs" "8" `
  "--batch-size" "32" `
  "--train-limit" "24000" `
  "--val-limit" "1800" `
  "--workers" "4" `
  "--seed" "3407"
exit $LASTEXITCODE
