# Creates the BIMA Desktop shortcut for this project's native Electron shell.
# The Electron runtime comes from this project's own node_modules (declared as
# a devDependency), so the shortcut has no dependency on any other folder.

$ErrorActionPreference = 'Stop'

$root     = Split-Path -Parent $PSScriptRoot
$mainJs   = Join-Path $PSScriptRoot 'main.cjs'
$appDir   = $PSScriptRoot
$electron = Join-Path $root 'node_modules\electron\dist\electron.exe'

if (-not (Test-Path $electron)) { throw "Electron not found at $electron. Run 'npm install' in $root first." }
if (-not (Test-Path $mainJs))   { throw "Shell entry not found at $mainJs" }

$link = Join-Path ([Environment]::GetFolderPath('Desktop')) 'BIMA.lnk'
$icon = Join-Path $root 'public\bima-desktop.ico'

$shellCom = New-Object -ComObject WScript.Shell
$shortcut = $shellCom.CreateShortcut($link)
$shortcut.TargetPath       = $electron
$shortcut.Arguments        = "`"$appDir`""
$shortcut.WorkingDirectory = $root
$shortcut.Description      = 'BIMA'
$shortcut.WindowStyle      = 1
if (Test-Path $icon) { $shortcut.IconLocation = "$icon,0" }
$shortcut.Save()

Write-Output "Shortcut now launches: $appDir"
Write-Output "Shortcut path: $link"
