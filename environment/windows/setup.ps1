$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$EnvFile = Join-Path $ScriptDir "environment.yml"
$ReqFile = Join-Path $ScriptDir "requirements.txt"
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..\\..")

if (-not (Test-Path -LiteralPath $EnvFile)) {
    throw "Missing environment spec: $EnvFile"
}

if (-not (Test-Path -LiteralPath $ReqFile)) {
    throw "Missing requirements file: $ReqFile"
}

$mambaCmd = $null
if (Get-Command micromamba -ErrorAction SilentlyContinue) {
    $mambaCmd = "micromamba"
} elseif (Get-Command mamba -ErrorAction SilentlyContinue) {
    $mambaCmd = "mamba"
} else {
    throw "Missing dependency: install micromamba or mamba."
}

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "Missing dependency: install uv."
}

$nameLine = Get-Content -LiteralPath $EnvFile |
    Where-Object { $_ -match '^\s*name\s*:' } |
    Select-Object -First 1

if (-not $nameLine) {
    throw "Could not parse environment name from $EnvFile. Expected 'name: <env_name>'."
}

$envName = ($nameLine -split ':', 2)[1].Trim()
if ([string]::IsNullOrWhiteSpace($envName)) {
    throw "Parsed environment name is empty in $EnvFile."
}

function Install-JsDependencies {
    $jsProjectRoots = @(
        $RepoRoot,
        (Join-Path $RepoRoot "illustrate_web\\frontend")
    )

    $foundProjects = 0

    foreach ($root in $jsProjectRoots) {
        $packageJson = Join-Path $root "package.json"
        if (-not (Test-Path -LiteralPath $packageJson)) {
            continue
        }

        $foundProjects += 1

        $pnpmLock = Join-Path $root "pnpm-lock.yaml"
        if (Test-Path -LiteralPath $pnpmLock) {
            & $mambaCmd "run" "-n" $envName "pnpm" "--version" *> $null
            if ($LASTEXITCODE -ne 0) {
                throw "Missing dependency in environment '$envName': pnpm (required by pnpm-lock.yaml)."
            }
            Push-Location $root
            try {
                & $mambaCmd "run" "-n" $envName "pnpm" "install" "--frozen-lockfile"
            } finally {
                Pop-Location
            }
            continue
        }

        $npmLock = Join-Path $root "package-lock.json"
        if (Test-Path -LiteralPath $npmLock) {
            & $mambaCmd "run" "-n" $envName "npm" "--version" *> $null
            if ($LASTEXITCODE -ne 0) {
                throw "Missing dependency in environment '$envName': npm (required by package-lock.json)."
            }
            Push-Location $root
            try {
                & $mambaCmd "run" "-n" $envName "npm" "ci"
            } finally {
                Pop-Location
            }
            continue
        }

        $yarnLock = Join-Path $root "yarn.lock"
        if (Test-Path -LiteralPath $yarnLock) {
            & $mambaCmd "run" "-n" $envName "yarn" "--version" *> $null
            if ($LASTEXITCODE -ne 0) {
                throw "Missing dependency in environment '$envName': yarn (required by yarn.lock)."
            }
            Push-Location $root
            try {
                & $mambaCmd "run" "-n" $envName "yarn" "install" "--frozen-lockfile"
            } finally {
                Pop-Location
            }
            continue
        }

        Write-Warning "Detected package.json without lockfile. Running npm install for initial lock generation."
        & $mambaCmd "run" "-n" $envName "npm" "--version" *> $null
        if ($LASTEXITCODE -ne 0) {
            throw "Missing dependency in environment '$envName': npm (required to install package.json dependencies)."
        }
        Push-Location $root
        try {
            & $mambaCmd "run" "-n" $envName "npm" "install"
        } finally {
            Pop-Location
        }
    }

    if ($foundProjects -eq 0) {
        Write-Host "No package.json found for JS dependency install."
    }
}

$hasEnv = $false
foreach ($line in (& $mambaCmd env list)) {
    if ($line -match "(?m)^\s*$([regex]::Escape($envName))\s") {
        $hasEnv = $true
        break
    }
}

if ($hasEnv) {
    & $mambaCmd "env" "update" "-n" $envName "-f" $EnvFile "--prune" "-y"
} else {
    & $mambaCmd "env" "create" "-n" $envName "-f" $EnvFile "-y"
}

& $mambaCmd "run" "-n" $envName "uv" "pip" "install" "-r" $ReqFile
Install-JsDependencies

Write-Host "Environment '$envName' is ready."
