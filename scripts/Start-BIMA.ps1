$ErrorActionPreference = 'Stop'

$projectRoot = Split-Path -Parent $PSScriptRoot
$mainJs = Join-Path $projectRoot 'desktop\main.cjs'
$appDir = Join-Path $projectRoot 'desktop'
$electron = 'C:\Users\mwstr\BIMA\node_modules\electron\dist\electron.exe'

if (-not (Test-Path -LiteralPath $mainJs)) { throw "Desktop shell not found: $mainJs" }
if (-not (Test-Path -LiteralPath $electron)) { throw "Electron runtime not found: $electron" }

Start-Process -FilePath $electron -ArgumentList "`"$appDir`"" -WorkingDirectory $projectRoot
