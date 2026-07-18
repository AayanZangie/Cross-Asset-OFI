param(
    [switch]$DenseRefit,
    [switch]$DeepLevels
)

$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Repo

New-Item -ItemType Directory -Force .\output\grid | Out-Null
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$Log = Join-Path $Repo "output\grid\search_log_$Stamp.log"
$env:PYTHONUNBUFFERED = "1"

$ArgsList = @(".\src\tests.py")
if ($DenseRefit) { $ArgsList += "--dense-refit" }
if ($DeepLevels) { $ArgsList += "--deep-levels" }

Write-Host "Cross-asset OFI grid started at $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host "Repo: $Repo"
Write-Host "Log:  $Log"
Write-Host "Args: python $($ArgsList -join ' ')"
Write-Host ""
Write-Host "The Python script prints progress, elapsed time, average time per config and ETA after every completed config."
Write-Host ""

python @ArgsList 2>&1 | Tee-Object -FilePath $Log -Append
$Code = $LASTEXITCODE

Write-Host ""
Write-Host "Finished at $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') with exit code $Code"
Write-Host "Log saved to $Log"
exit $Code
