param(
    [ValidateSet("onefile", "onedir")]
    [string]$Mode = "onefile"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repoRoot

Write-Host "[build] repo root: $repoRoot"

if ($env:VIRTUAL_ENV) {
    $pythonExe = Join-Path $env:VIRTUAL_ENV "Scripts\python.exe"
    if (-not (Test-Path $pythonExe)) {
        throw "VIRTUAL_ENV is set but python.exe was not found: $pythonExe"
    }
}
elseif ($env:CONDA_PREFIX) {
    $pythonExe = Join-Path $env:CONDA_PREFIX "python.exe"
    if (-not (Test-Path $pythonExe)) {
        throw "CONDA_PREFIX is set but python.exe was not found: $pythonExe"
    }
}
else {
    $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
    if ($null -eq $pythonCmd) {
        throw "python was not found. Please activate your virtual environment first."
    }
    $pythonExe = $pythonCmd.Source
}

Write-Host "[build] python: $pythonExe"

& $pythonExe -m PyInstaller --version | Out-Host
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller is not available in current interpreter. Install with: python -m pip install pyinstaller"
}

& $pythonExe -c "import ultralytics, torch; print('ok')" | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "Missing required packages for distributable build. Please install ultralytics and torch first."
}

function Clear-BuildFolder {
    param(
        [Parameter(Mandatory = $true)]
        [string]$TargetPath
    )

    if (-not (Test-Path $TargetPath)) {
        return
    }

    Write-Host "[build] clean: $TargetPath"
    try {
        Remove-Item -LiteralPath $TargetPath -Recurse -Force -ErrorAction Stop
    }
    catch {
        Write-Warning "Failed to fully remove $TargetPath. Continue build and let PyInstaller --clean handle remaining files."
    }
}

foreach ($folder in @("build", "dist")) {
    $target = Join-Path $repoRoot $folder
    Clear-BuildFolder -TargetPath $target
}

$specFile = Join-Path $repoRoot "uestc4006p_gui.spec"
if (-not (Test-Path $specFile)) {
    throw "Spec file not found: $specFile"
}

$tempRoot = Join-Path $env:TEMP "uestc4006p_gui_pyinstaller"
$tempDist = Join-Path $tempRoot "dist"
$tempWork = Join-Path $tempRoot "build"
Clear-BuildFolder -TargetPath $tempRoot
New-Item -ItemType Directory -Path $tempRoot -Force | Out-Null

Write-Host "[build] run: PyInstaller $Mode"
if ($Mode -eq "onefile") {
    & $pythonExe -m PyInstaller --noconfirm --clean --distpath $tempDist --workpath $tempWork $specFile
}
else {
    $entryScript = Join-Path $repoRoot "src\\uestc4006p_gui\\app.py"
    $srcPath = Join-Path $repoRoot "src"

    & $pythonExe -m PyInstaller `
        --noconfirm --clean `
        --onedir `
        --name "uestc4006p_gui" `
        --paths $srcPath `
        --collect-all cv2 `
        --collect-all ultralytics `
        --collect-all torch `
        --collect-submodules ultralytics `
        --collect-submodules torch `
        --exclude-module PyQt5 `
        --exclude-module PyQt6 `
        --distpath $tempDist `
        --workpath $tempWork `
        $entryScript
}
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller build failed."
}

$distDir = Join-Path $repoRoot "dist"
New-Item -ItemType Directory -Path $distDir -Force | Out-Null

if ($Mode -eq "onedir") {
    $tempDir = Join-Path $tempDist "uestc4006p_gui"
    if (-not (Test-Path $tempDir)) {
        throw "Build finished but temp onedir was not found: $tempDir"
    }
    $targetDir = Join-Path $distDir "uestc4006p_gui"
    Clear-BuildFolder -TargetPath $targetDir
    if (-not (Test-Path $targetDir)) {
        New-Item -ItemType Directory -Path $targetDir -Force | Out-Null
    }
    Get-ChildItem -LiteralPath $tempDir -Force | ForEach-Object {
        Copy-Item -LiteralPath $_.FullName -Destination $targetDir -Recurse -Force
    }
    Write-Host "[build] success: $targetDir"
    return
}

$tempExe = Join-Path $tempDist "uestc4006p_gui.exe"
if (-not (Test-Path $tempExe)) {
    throw "Build finished but temp exe was not found: $tempExe"
}

$exePath = Join-Path $distDir "uestc4006p_gui.exe"
if (Test-Path $exePath) {
    try {
        Remove-Item -LiteralPath $exePath -Force -ErrorAction Stop
    }
    catch {
        $fallbackExe = Join-Path $distDir "uestc4006p_gui.new.exe"
        Copy-Item -LiteralPath $tempExe -Destination $fallbackExe -Force
        Write-Warning "Old exe is locked and cannot be replaced: $exePath"
        Write-Host "[build] success (fallback): $fallbackExe"
        return
    }
}

Copy-Item -LiteralPath $tempExe -Destination $exePath -Force

Write-Host "[build] success: $exePath"
