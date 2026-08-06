param([int]$WaitForPid = 0)
$ErrorActionPreference = "Stop"
if ($WaitForPid -gt 0 -and (Get-Process -Id $WaitForPid -ErrorAction SilentlyContinue)) {
  Wait-Process -Id $WaitForPid
}

& python "benchmark_adult_pose.py" "--model" "vitpose" "--batch-size" "24" "--output" "runs/baseline-vitpose-full.json"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& python "benchmark_adult_pose.py" "--model" "kinepose" "--checkpoint" "runs/ablation-plain/best.pt" "--batch-size" "32" "--output" "runs/benchmark-plain-full.json"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& python "benchmark_adult_pose.py" "--model" "kinepose" "--checkpoint" "runs/ablation-hybrid/best.pt" "--batch-size" "32" "--output" "runs/benchmark-hybrid-full.json"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& python "benchmark_adult_pose.py" "--model" "kinepose" "--checkpoint" "runs/ablation-hybrid-sharp/best.pt" "--batch-size" "32" "--output" "runs/benchmark-hybrid-sharp-full.json"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& python "benchmark_adult_pose.py" "--model" "kinepose" "--checkpoint" "runs/ablation-hybrid-anchor/best.pt" "--batch-size" "32" "--output" "runs/benchmark-hybrid-anchor-full.json"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& python "benchmark_adult_pose.py" "--model" "kinepose" "--checkpoint" "runs/ablation-hybrid-fast/best.pt" "--batch-size" "32" "--output" "runs/benchmark-hybrid-fast-full.json"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& python "benchmark_adult_pose.py" "--model" "simplebaseline" "--checkpoint" "runs/baseline-simple-r18/best.pt" "--batch-size" "24" "--output" "runs/benchmark-simple-r18-full.json"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& python "benchmark_robustness.py" "--model" "vitpose" "--limit" "1000" "--batch-size" "24" "--output" "runs/robustness-vitpose.json"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& python "benchmark_robustness.py" "--model" "kinepose" "--checkpoint" "runs/ablation-plain/best.pt" "--limit" "1000" "--batch-size" "32" "--output" "runs/robustness-plain.json"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& python "benchmark_robustness.py" "--model" "kinepose" "--checkpoint" "runs/ablation-hybrid/best.pt" "--limit" "1000" "--batch-size" "32" "--output" "runs/robustness-hybrid.json"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& python "benchmark_robustness.py" "--model" "kinepose" "--checkpoint" "runs/ablation-hybrid-sharp/best.pt" "--limit" "1000" "--batch-size" "32" "--output" "runs/robustness-hybrid-sharp.json"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& python "benchmark_robustness.py" "--model" "kinepose" "--checkpoint" "runs/ablation-hybrid-anchor/best.pt" "--limit" "1000" "--batch-size" "32" "--output" "runs/robustness-hybrid-anchor.json"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& python "benchmark_robustness.py" "--model" "kinepose" "--checkpoint" "runs/ablation-hybrid-fast/best.pt" "--limit" "1000" "--batch-size" "32" "--output" "runs/robustness-hybrid-fast.json"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& python "benchmark_robustness.py" "--model" "simplebaseline" "--checkpoint" "runs/baseline-simple-r18/best.pt" "--limit" "1000" "--batch-size" "24" "--output" "runs/robustness-simple-r18.json"
exit $LASTEXITCODE
