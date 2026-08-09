param(
    [ValidateSet("ModelScope", "HuggingFace")]
    [string]$Provider = "ModelScope",
    [string]$ModelId,
    [string]$ModelDir,
    [switch]$SkipDependencyInstall
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if ([string]::IsNullOrWhiteSpace($ModelId)) {
    $ModelId = if ($env:ASR_MODEL_ID) { $env:ASR_MODEL_ID } else { "Qwen/Qwen3-ASR-1.7B" }
}

if ([string]::IsNullOrWhiteSpace($ModelDir)) {
    $ModelDir = if ($env:ASR_LOCAL_MODEL_DIR) {
        $env:ASR_LOCAL_MODEL_DIR
    } else {
        Join-Path $PSScriptRoot "models\Qwen3-ASR-1.7B"
    }
}
$Python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    throw ".venv was not found. Run setup.bat first."
}

function Test-LocalModelReady {
    param([string]$Path)
    if (-not (Test-Path (Join-Path $Path "config.json"))) {
        return $false
    }
    $weights = Get-ChildItem -LiteralPath $Path -File -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -like "*.safetensors" -or $_.Name -like "*.bin" } |
        Select-Object -First 1
    return $null -ne $weights
}

if (Test-LocalModelReady $ModelDir) {
    Write-Host "Model already exists: $ModelDir"
    return
}

New-Item -ItemType Directory -Force -Path $ModelDir | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $PSScriptRoot ".cache\modelscope") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $PSScriptRoot ".cache\huggingface") | Out-Null

$env:MODELSCOPE_CACHE = Join-Path $PSScriptRoot ".cache\modelscope"
$env:HF_HOME = Join-Path $PSScriptRoot ".cache\huggingface"

if ($Provider -eq "ModelScope") {
    if (-not $SkipDependencyInstall) {
        & $Python -m pip install --upgrade modelscope
        if ($LASTEXITCODE -ne 0) { throw "Failed to install/upgrade modelscope." }
    }

    $ModelScopeExe = Join-Path $PSScriptRoot ".venv\Scripts\modelscope.exe"
    if (-not (Test-Path $ModelScopeExe)) {
        throw "modelscope.exe was not found after installation."
    }

    Write-Host ""
    Write-Host "Downloading $ModelId from ModelScope ..."
    Write-Host "Target: $ModelDir"
    & $ModelScopeExe download --model $ModelId --local_dir $ModelDir
    if ($LASTEXITCODE -ne 0) {
        throw "ModelScope download failed with exit code $LASTEXITCODE."
    }
}
else {
    if (-not $SkipDependencyInstall) {
        & $Python -m pip install --upgrade "huggingface_hub[cli]"
        if ($LASTEXITCODE -ne 0) { throw "Failed to install/upgrade huggingface_hub." }
    }

    $HfExe = Join-Path $PSScriptRoot ".venv\Scripts\hf.exe"
    if (-not (Test-Path $HfExe)) {
        throw "hf.exe was not found after installation."
    }

    Write-Host ""
    Write-Host "Downloading $ModelId from Hugging Face ..."
    Write-Host "Target: $ModelDir"
    & $HfExe download $ModelId --local-dir $ModelDir
    if ($LASTEXITCODE -ne 0) {
        throw "Hugging Face download failed with exit code $LASTEXITCODE."
    }
}

if (-not (Test-LocalModelReady $ModelDir)) {
    throw "Download command finished, but the local model does not look complete: $ModelDir"
}

Write-Host ""
Write-Host "Model ready: $ModelDir"
Write-Host "The service will load this local checkpoint when its matching model option is enabled."
