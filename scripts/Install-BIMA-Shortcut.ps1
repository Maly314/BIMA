$ErrorActionPreference = 'Stop'

$projectRoot = Split-Path -Parent $PSScriptRoot
$mainJs = Join-Path $projectRoot 'desktop\main.cjs'
$appDir = Join-Path $projectRoot 'desktop'
$electron = 'C:\Users\mwstr\BIMA\node_modules\electron\dist\electron.exe'
$icon = Join-Path $projectRoot 'public\bima-desktop.ico'
$desktop = [Environment]::GetFolderPath('Desktop')
$link = Join-Path $desktop 'BIMA.lnk'

if (-not (Test-Path -LiteralPath $mainJs)) { throw "Desktop shell not found: $mainJs" }
if (-not (Test-Path -LiteralPath $electron)) { throw "Electron runtime not found: $electron" }
if (-not (Test-Path -LiteralPath $icon)) { throw "BIMA icon not found: $icon" }

$shell = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut($link)
$shortcut.TargetPath = $electron
$shortcut.Arguments = "`"$appDir`""
$shortcut.WorkingDirectory = $projectRoot
$shortcut.IconLocation = "$icon,0"
$shortcut.Description = 'BIMA'
$shortcut.WindowStyle = 7
$shortcut.Save()

Write-Output "Shortcut created: $link"
