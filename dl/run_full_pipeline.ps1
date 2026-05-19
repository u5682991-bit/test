param(
  [string]$Device = "cuda",
  [switch]$Amp,
  [int]$Epochs = 20,
  [int]$BatchSize = 32,
  [int]$NumWorkers = 8,
  [switch]$SkipRegistration,
  [switch]$SkipExisting
)

$ErrorActionPreference = "Stop"
Set-Location (Resolve-Path "$PSScriptRoot\..")

$argsList = @(
  "dl\run_full_pipeline.py",
  "--device", $Device,
  "--epochs", "$Epochs",
  "--batch-size", "$BatchSize",
  "--num-workers", "$NumWorkers"
)

if ($Amp) { $argsList += "--amp" }
if ($SkipRegistration) { $argsList += "--skip-registration" }
if ($SkipExisting) { $argsList += "--skip-existing" }

python @argsList

