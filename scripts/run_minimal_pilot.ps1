[CmdletBinding()]
param(
    [string]$EtlSar = "etl-sar",
    [string]$RunRoot,
    [string]$LegacyReference,
    [switch]$WhatIf
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($RunRoot)) {
    $RunRoot = Join-Path $projectRoot "runs/minimal_pilot"
}

$handConfig = Join-Path $projectRoot "configs/hand_quick.yaml"
$legConfig = Join-Path $projectRoot "configs/leg_quick.yaml"
$handRoot = Join-Path $RunRoot "hand"
$legRoot = Join-Path $RunRoot "leg"

function Get-StageSignature {
    param(
        [string]$Domain,
        [string]$ConfigPath,
        [string]$Budgets,
        [int]$Seed,
        [string]$CommandText
    )

    $configHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $ConfigPath).Hash
    $payload = @{
        domain = $Domain
        config_sha256 = $configHash
        budgets = $Budgets
        seed = $Seed
        command = $CommandText
    } | ConvertTo-Json -Compress
    $bytes = [System.Text.Encoding]::UTF8.GetBytes($payload)
    $sha256 = [System.Security.Cryptography.SHA256]::Create()
    try {
        return [System.BitConverter]::ToString($sha256.ComputeHash($bytes)).Replace("-", "").ToLowerInvariant()
    }
    finally {
        $sha256.Dispose()
    }
}

function Test-StageComplete {
    param(
        [string]$MarkerPath,
        [string]$Signature,
        [string[]]$ExpectedArtifacts
    )

    if (-not (Test-Path -LiteralPath $MarkerPath -PathType Leaf)) {
        return $false
    }
    try {
        $marker = Get-Content -Raw -Encoding UTF8 -LiteralPath $MarkerPath | ConvertFrom-Json
    }
    catch {
        return $false
    }
    if ($marker.signature -ne $Signature) {
        return $false
    }
    foreach ($artifact in $ExpectedArtifacts) {
        if (-not (Test-Path -LiteralPath $artifact)) {
            return $false
        }
    }
    return $true
}

function Invoke-Stage {
    param(
        [string]$Label,
        [string]$Domain,
        [string]$ConfigPath,
        [string]$Budgets,
        [int]$Seed,
        [string]$CommandText,
        [string]$StageDirectory,
        [string[]]$ExpectedArtifacts,
        [scriptblock]$Action
    )

    $signature = Get-StageSignature -Domain $Domain -ConfigPath $ConfigPath `
        -Budgets $Budgets -Seed $Seed -CommandText $CommandText
    if ($WhatIf) {
        Write-Output "[WhatIf] ${Label}: $CommandText"
        return
    }

    $markerPath = Join-Path $StageDirectory "stage.complete.json"
    if (Test-StageComplete -MarkerPath $markerPath -Signature $signature `
            -ExpectedArtifacts $ExpectedArtifacts) {
        Write-Output "[Skip] $Label"
        return
    }

    New-Item -ItemType Directory -Force -Path $StageDirectory | Out-Null
    Write-Output "[Run] $Label"
    & $Action
    if ($null -ne $LASTEXITCODE -and $LASTEXITCODE -ne 0) {
        throw "$Label failed with exit code $LASTEXITCODE"
    }
    foreach ($artifact in $ExpectedArtifacts) {
        if (-not (Test-Path -LiteralPath $artifact)) {
            throw "$Label did not create expected artifact: $artifact"
        }
    }
    @{
        signature = $signature
        label = $Label
        completed_utc = [DateTime]::UtcNow.ToString("o")
        expected_artifacts = $ExpectedArtifacts
    } | ConvertTo-Json -Depth 4 | Set-Content -Encoding UTF8 -LiteralPath $markerPath
}

$handBudgets = "source_min=10000;source_max=30000;success_actions=20;sar_steps=200;target_steps=20000;freeze_steps=2000;eval_freq=5000;episodes=10"
$legBudgets = "source_min=20000;source_max=50000;success_actions=20;sar_steps=200;target_steps=20000;freeze_steps=2000;eval_freq=5000;episodes=10"

$handSource = Join-Path $handRoot "source"
$handRepresentation = Join-Path $handRoot "representation"
$handBaseline = Join-Path $handRoot "baseline_target"
$handExtension = Join-Path $handRoot "extension_target"
$handBaselineEval = Join-Path $handRoot "baseline_eval"
$handExtensionEval = Join-Path $handRoot "extension_eval"
$handComparison = Join-Path $handRoot "comparison.json"
$handBundle = Join-Path $handRepresentation "representation_bundle.pt"

Invoke-Stage -Label "hand/source" -Domain "hand" -ConfigPath $handConfig `
    -Budgets $handBudgets -Seed 7 `
    -CommandText "$EtlSar explore --config $handConfig --run-dir $handSource --timesteps 30000 --min-timesteps 10000 --min-success-actions 20" `
    -StageDirectory $handSource `
    -ExpectedArtifacts @((Join-Path $handSource "latest_explorer.zip"), (Join-Path $handSource "representation.pt"), (Join-Path $handSource "data")) `
    -Action {
        & $EtlSar explore --config $handConfig --run-dir $handSource `
            --timesteps 30000 --min-timesteps 10000 --min-success-actions 20
    }

Invoke-Stage -Label "hand/representation" -Domain "hand" -ConfigPath $handConfig `
    -Budgets $handBudgets -Seed 7 `
    -CommandText "$EtlSar fit-representation --config $handConfig --data-dir $handSource/data --explore-checkpoint $handSource/representation.pt --output-dir $handRepresentation --sar-steps 200" `
    -StageDirectory $handRepresentation `
    -ExpectedArtifacts @($handBundle, (Join-Path $handRepresentation "synergy.joblib")) `
    -Action {
        & $EtlSar fit-representation --config $handConfig `
            --data-dir (Join-Path $handSource "data") `
            --explore-checkpoint (Join-Path $handSource "representation.pt") `
            --output-dir $handRepresentation --sar-steps 200
    }

Invoke-Stage -Label "hand/baseline" -Domain "hand" -ConfigPath $handConfig `
    -Budgets $handBudgets -Seed 7 `
    -CommandText "$EtlSar transfer --config $handConfig --bundle $handBundle --run-dir $handBaseline --timesteps 20000 --decoder-freeze-steps 2000 --eval-freq 5000 --sar-scale 0.0" `
    -StageDirectory $handBaseline `
    -ExpectedArtifacts @((Join-Path $handBaseline "latest_model.zip")) `
    -Action {
        & $EtlSar transfer --config $handConfig --bundle $handBundle `
            --run-dir $handBaseline --timesteps 20000 `
            --decoder-freeze-steps 2000 --eval-freq 5000 --sar-scale 0.0
    }

Invoke-Stage -Label "hand/extension" -Domain "hand" -ConfigPath $handConfig `
    -Budgets $handBudgets -Seed 7 `
    -CommandText "$EtlSar transfer --config $handConfig --bundle $handBundle --run-dir $handExtension --timesteps 20000 --decoder-freeze-steps 2000 --eval-freq 5000 --sar-scale 1.0" `
    -StageDirectory $handExtension `
    -ExpectedArtifacts @((Join-Path $handExtension "latest_model.zip")) `
    -Action {
        & $EtlSar transfer --config $handConfig --bundle $handBundle `
            --run-dir $handExtension --timesteps 20000 `
            --decoder-freeze-steps 2000 --eval-freq 5000 --sar-scale 1.0
    }

Invoke-Stage -Label "hand/baseline-evaluation" -Domain "hand" -ConfigPath $handConfig `
    -Budgets $handBudgets -Seed 7 `
    -CommandText "$EtlSar evaluate --config $handConfig --bundle $handBundle --model-path $handBaseline/latest_model.zip --output-dir $handBaselineEval --episodes 10 --environment-steps 20000 --sar-scale 0.0" `
    -StageDirectory $handBaselineEval `
    -ExpectedArtifacts @((Join-Path $handBaselineEval "summary.json"), (Join-Path $handBaselineEval "episodes.csv")) `
    -Action {
        & $EtlSar evaluate --config $handConfig --bundle $handBundle `
            --model-path (Join-Path $handBaseline "latest_model.zip") `
            --output-dir $handBaselineEval --episodes 10 `
            --environment-steps 20000 --sar-scale 0.0
    }

Invoke-Stage -Label "hand/extension-evaluation" -Domain "hand" -ConfigPath $handConfig `
    -Budgets $handBudgets -Seed 7 `
    -CommandText "$EtlSar evaluate --config $handConfig --bundle $handBundle --model-path $handExtension/latest_model.zip --output-dir $handExtensionEval --episodes 10 --environment-steps 20000 --sar-scale 1.0" `
    -StageDirectory $handExtensionEval `
    -ExpectedArtifacts @((Join-Path $handExtensionEval "summary.json"), (Join-Path $handExtensionEval "episodes.csv")) `
    -Action {
        & $EtlSar evaluate --config $handConfig --bundle $handBundle `
            --model-path (Join-Path $handExtension "latest_model.zip") `
            --output-dir $handExtensionEval --episodes 10 `
            --environment-steps 20000 --sar-scale 1.0
    }

Invoke-Stage -Label "hand/comparison" -Domain "hand" -ConfigPath $handConfig `
    -Budgets $handBudgets -Seed 7 `
    -CommandText "$EtlSar compare --baseline $handBaselineEval/summary.json --extension $handExtensionEval/summary.json --output $handComparison" `
    -StageDirectory (Join-Path $handRoot ".comparison-stage") `
    -ExpectedArtifacts @($handComparison) `
    -Action {
        & $EtlSar compare --baseline (Join-Path $handBaselineEval "summary.json") `
            --extension (Join-Path $handExtensionEval "summary.json") `
            --output $handComparison
    }

$legSource = Join-Path $legRoot "source"
$legRepresentation = Join-Path $legRoot "representation"
$legBaseline = Join-Path $legRoot "baseline_target"
$legExtension = Join-Path $legRoot "extension_target"
$legBaselineEval = Join-Path $legRoot "baseline_eval"
$legExtensionEval = Join-Path $legRoot "extension_eval"
$legComparison = Join-Path $legRoot "comparison.json"
$legBundle = Join-Path $legRepresentation "representation_bundle.pt"

Invoke-Stage -Label "leg/source" -Domain "leg" -ConfigPath $legConfig `
    -Budgets $legBudgets -Seed 11 `
    -CommandText "$EtlSar explore --config $legConfig --run-dir $legSource --timesteps 50000 --min-timesteps 20000 --min-success-actions 20" `
    -StageDirectory $legSource `
    -ExpectedArtifacts @((Join-Path $legSource "latest_explorer.zip"), (Join-Path $legSource "representation.pt"), (Join-Path $legSource "data")) `
    -Action {
        & $EtlSar explore --config $legConfig --run-dir $legSource `
            --timesteps 50000 --min-timesteps 20000 --min-success-actions 20
    }

Invoke-Stage -Label "leg/representation" -Domain "leg" -ConfigPath $legConfig `
    -Budgets $legBudgets -Seed 11 `
    -CommandText "$EtlSar fit-representation --config $legConfig --data-dir $legSource/data --explore-checkpoint $legSource/representation.pt --output-dir $legRepresentation --sar-steps 200" `
    -StageDirectory $legRepresentation `
    -ExpectedArtifacts @($legBundle, (Join-Path $legRepresentation "synergy.joblib")) `
    -Action {
        & $EtlSar fit-representation --config $legConfig `
            --data-dir (Join-Path $legSource "data") `
            --explore-checkpoint (Join-Path $legSource "representation.pt") `
            --output-dir $legRepresentation --sar-steps 200
    }

Invoke-Stage -Label "leg/baseline" -Domain "leg" -ConfigPath $legConfig `
    -Budgets $legBudgets -Seed 11 `
    -CommandText "$EtlSar transfer --config $legConfig --bundle $legBundle --run-dir $legBaseline --timesteps 20000 --decoder-freeze-steps 2000 --eval-freq 5000 --sar-scale 0.0" `
    -StageDirectory $legBaseline `
    -ExpectedArtifacts @((Join-Path $legBaseline "latest_model.zip")) `
    -Action {
        & $EtlSar transfer --config $legConfig --bundle $legBundle `
            --run-dir $legBaseline --timesteps 20000 `
            --decoder-freeze-steps 2000 --eval-freq 5000 --sar-scale 0.0
    }

Invoke-Stage -Label "leg/extension" -Domain "leg" -ConfigPath $legConfig `
    -Budgets $legBudgets -Seed 11 `
    -CommandText "$EtlSar transfer --config $legConfig --bundle $legBundle --run-dir $legExtension --timesteps 20000 --decoder-freeze-steps 2000 --eval-freq 5000 --sar-scale 1.0" `
    -StageDirectory $legExtension `
    -ExpectedArtifacts @((Join-Path $legExtension "latest_model.zip")) `
    -Action {
        & $EtlSar transfer --config $legConfig --bundle $legBundle `
            --run-dir $legExtension --timesteps 20000 `
            --decoder-freeze-steps 2000 --eval-freq 5000 --sar-scale 1.0
    }

Invoke-Stage -Label "leg/baseline-evaluation" -Domain "leg" -ConfigPath $legConfig `
    -Budgets $legBudgets -Seed 11 `
    -CommandText "$EtlSar evaluate --config $legConfig --bundle $legBundle --model-path $legBaseline/latest_model.zip --output-dir $legBaselineEval --episodes 10 --environment-steps 20000 --sar-scale 0.0" `
    -StageDirectory $legBaselineEval `
    -ExpectedArtifacts @((Join-Path $legBaselineEval "summary.json"), (Join-Path $legBaselineEval "episodes.csv")) `
    -Action {
        & $EtlSar evaluate --config $legConfig --bundle $legBundle `
            --model-path (Join-Path $legBaseline "latest_model.zip") `
            --output-dir $legBaselineEval --episodes 10 `
            --environment-steps 20000 --sar-scale 0.0
    }

Invoke-Stage -Label "leg/extension-evaluation" -Domain "leg" -ConfigPath $legConfig `
    -Budgets $legBudgets -Seed 11 `
    -CommandText "$EtlSar evaluate --config $legConfig --bundle $legBundle --model-path $legExtension/latest_model.zip --output-dir $legExtensionEval --episodes 10 --environment-steps 20000 --sar-scale 1.0" `
    -StageDirectory $legExtensionEval `
    -ExpectedArtifacts @((Join-Path $legExtensionEval "summary.json"), (Join-Path $legExtensionEval "episodes.csv")) `
    -Action {
        & $EtlSar evaluate --config $legConfig --bundle $legBundle `
            --model-path (Join-Path $legExtension "latest_model.zip") `
            --output-dir $legExtensionEval --episodes 10 `
            --environment-steps 20000 --sar-scale 1.0
    }

Invoke-Stage -Label "leg/comparison" -Domain "leg" -ConfigPath $legConfig `
    -Budgets $legBudgets -Seed 11 `
    -CommandText "$EtlSar compare --baseline $legBaselineEval/summary.json --extension $legExtensionEval/summary.json --output $legComparison" `
    -StageDirectory (Join-Path $legRoot ".comparison-stage") `
    -ExpectedArtifacts @($legComparison) `
    -Action {
        & $EtlSar compare --baseline (Join-Path $legBaselineEval "summary.json") `
            --extension (Join-Path $legExtensionEval "summary.json") `
            --output $legComparison
    }

$pilotSummary = Join-Path $RunRoot "pilot_summary.json"
$summaryCommand = "$EtlSar pilot-summary --hand $handComparison --leg $legComparison --output $pilotSummary"
if (-not [string]::IsNullOrWhiteSpace($LegacyReference)) {
    $summaryCommand += " --legacy-reference $LegacyReference"
}
Invoke-Stage -Label "pilot-summary" -Domain "hand+leg" -ConfigPath $handConfig `
    -Budgets "$handBudgets|$legBudgets" -Seed 7 `
    -CommandText $summaryCommand `
    -StageDirectory (Join-Path $RunRoot ".summary-stage") `
    -ExpectedArtifacts @($pilotSummary) `
    -Action {
        if ([string]::IsNullOrWhiteSpace($LegacyReference)) {
            & $EtlSar pilot-summary --hand $handComparison --leg $legComparison `
                --output $pilotSummary
        }
        else {
            & $EtlSar pilot-summary --hand $handComparison --leg $legComparison `
                --output $pilotSummary --legacy-reference $LegacyReference
        }
    }
