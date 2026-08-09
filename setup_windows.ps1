$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    py -3.12 -m venv .venv
}

$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
& $python -m pip install --upgrade pip setuptools wheel

# Install CUDA-enabled PyTorch first. Change cu128 if your driver/environment needs another official wheel channel.
& $python -m pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
& $python -m pip install -r requirements_windows.txt

# Qwen officially recommends ModelScope for users in Mainland China. Install it
# separately so the ASR runtime dependencies stay simple.
& $python -m pip install --upgrade modelscope

& $python check_env.py
& $python test_itn.py

Write-Host ""
Write-Host "Downloading the ASR checkpoint from ModelScope into .\models\Qwen3-ASR-1.7B ..."
try {
    & "$PSScriptRoot\download_model.ps1" -Provider ModelScope -SkipDependencyInstall
    if ($LASTEXITCODE -ne 0) {
        throw "download_model.ps1 exited with code $LASTEXITCODE"
    }
}
catch {
    Write-Warning "ModelScope download did not complete: $($_.Exception.Message)"
    Write-Warning "Setup will continue. You can retry later with download_model.bat."
    Write-Warning "If no complete local model is present, server.py falls back to the Hugging Face model ID at startup."
}

Write-Host ""
Write-Host "Downloading the ForcedAligner checkpoint from ModelScope into .\models\Qwen3-ForcedAligner-0.6B ..."
try {
    & "$PSScriptRoot\download_model.ps1" -Provider ModelScope -SkipDependencyInstall `
        -ModelId "Qwen/Qwen3-ForcedAligner-0.6B" `
        -ModelDir (Join-Path $PSScriptRoot "models\Qwen3-ForcedAligner-0.6B")
    if ($LASTEXITCODE -ne 0) {
        throw "download_model.ps1 exited with code $LASTEXITCODE"
    }
}
catch {
    Write-Warning "ForcedAligner download did not complete: $($_.Exception.Message)"
    Write-Warning "Setup will continue. You can retry later with download_model.bat."
    Write-Warning "Timestamp requests will load the Hugging Face model ID if a local aligner checkpoint is unavailable."
}

Write-Host ""
Write-Host "Setup complete. Run start.bat"
