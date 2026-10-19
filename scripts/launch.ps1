$ErrorActionPreference = 'Stop'
$LabRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $LabRoot
$env:TOKENIZER_LAB_OFFLINE = '1'
$env:HF_HUB_OFFLINE = '1'
$env:HF_DATASETS_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:PYTHONNOUSERSITE = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:TOKENIZERS_PARALLELISM = 'false'
$env:PYTHONPATH = "$LabRoot\src;$LabRoot\vendor"
$LabPython = Join-Path $LabRoot 'runtime\python.exe'
if (-not (Test-Path -LiteralPath $LabPython)) { throw 'Extract the complete Windows offline kit before launching.' }
& $LabPython -m tokenizer_lab.cli launch @args
exit $LASTEXITCODE
