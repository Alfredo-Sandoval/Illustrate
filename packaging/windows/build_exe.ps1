$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..\..")

$AppName = if ($env:APP_NAME) { $env:APP_NAME } else { "Illustrate" }
$Version = if ($env:VERSION) {
    $env:VERSION
} else {
    python -c "import tomllib, pathlib; p = tomllib.loads(pathlib.Path('pyproject.toml').read_text(encoding='utf-8')); print(p.get('project', {}).get('version', '0.0.0'))"
}

$BuildRoot = Join-Path $RepoRoot "dist\package\windows"
$WorkPath = Join-Path $BuildRoot "build"
$DistPath = Join-Path $BuildRoot "dist"
$SpecPath = Join-Path $BuildRoot "spec"
$BundlePath = Join-Path $DistPath $AppName
$ZipPath = Join-Path $BuildRoot "$AppName-$Version-windows-x64.zip"

if (-not (Get-Command pyinstaller -ErrorAction SilentlyContinue)) {
    if (Get-Command uv -ErrorAction SilentlyContinue) {
        Write-Host "[package] Installing pyinstaller via uv..."
        & uv pip install pyinstaller
    } else {
        throw "Missing dependency: pyinstaller (install with 'uv pip install pyinstaller')."
    }
}

if (Test-Path -LiteralPath $BuildRoot) {
    Remove-Item -Recurse -Force $BuildRoot
}
New-Item -ItemType Directory -Force -Path $WorkPath, $DistPath, $SpecPath | Out-Null

Set-Location $RepoRoot

Write-Host "[package] Building $AppName.exe ($Version)..."
& pyinstaller `
    --noconfirm `
    --clean `
    --windowed `
    --name $AppName `
    --add-data "data;data" `
    --collect-submodules "illustrate_gui" `
    --workpath $WorkPath `
    --distpath $DistPath `
    --specpath $SpecPath `
    "illustrate_gui/main.py"

if (-not (Test-Path -LiteralPath $BundlePath)) {
    throw "Expected bundle not found: $BundlePath"
}

if (Test-Path -LiteralPath $ZipPath) {
    Remove-Item -Force $ZipPath
}

Write-Host "[package] Creating ZIP..."
Compress-Archive -Path (Join-Path $BundlePath "*") -DestinationPath $ZipPath -Force

Write-Host "[package] Done."
Write-Host "  Bundle: $BundlePath"
Write-Host "  ZIP:    $ZipPath"
