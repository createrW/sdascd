param(
    [Parameter(Mandatory=$true)]
    [ValidatePattern('^[A-Za-z0-9_-]+$')]
    [string]$RunId,
    [string]$FeaturesDir = '',
    [switch]$Reextract,
    [string]$PythonExe = 'python'
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
if (-not $env:MEMOTION_DATA_DIR) { throw 'Set MEMOTION_DATA_DIR to the dataset root before running.' }
if (-not (Test-Path -LiteralPath (Join-Path $env:MEMOTION_DATA_DIR 'train/metadata.csv'))) { throw 'MEMOTION_DATA_DIR must contain train/metadata.csv.' }
if ([bool]$FeaturesDir -eq [bool]$Reextract) { throw 'Choose exactly one: -FeaturesDir for a complete cache, or -Reextract.' }
$pythonCommand = Get-Command $PythonExe -ErrorAction Stop
$pythonExePath = $pythonCommand.Source
Push-Location $projectRoot
try {
    $featureDir = if ($Reextract) { "features/$RunId" } else { $FeaturesDir }
    $evidenceDir = "outputs/reproduction/$RunId"
    $targets = @($evidenceDir, "outputs/image/$RunId", "checkpoints/image/$RunId")
    if ($Reextract) { $targets += $featureDir }
    foreach ($target in $targets) {
        if (Test-Path -LiteralPath $target) { throw "Existing output: $target. Choose a new RunId." }
    }
    if (-not $Reextract -and -not (Test-Path -LiteralPath $featureDir -PathType Container)) { throw "Missing cache: $featureDir" }
    if (-not $env:HF_HOME) { $env:HF_HOME = Join-Path $projectRoot 'hf_cache' }
    $env:PYTHONUTF8 = '1'
    $referenceMetrics = 'outputs/image/20261006_code_fixes_validation_v2/metrics_validation.json'
    function Invoke-ReproductionStep {
        param([string]$Name, [string[]]$StepArgs)
        & $pythonExePath scripts/run_logged.py --log-dir "$evidenceDir/logs" --name $Name -- python @StepArgs
        if ($LASTEXITCODE -ne 0) { throw "Step $Name failed; inspect its log." }
    }
    Invoke-ReproductionStep '01_environment' @('-m','scripts.show_environment')
    Invoke-ReproductionStep '02_packages' @('-m','pip','freeze')
    if ($Reextract) {
        Invoke-ReproductionStep '03_dataset' @('-m','scripts.verify_dataset')
    } else {
        Invoke-ReproductionStep '03_dataset' @('-m','scripts.verify_dataset','--skip_images')
    }
    if ($Reextract) {
        Invoke-ReproductionStep '04_features' @('-m','src.image.extract_features','--splits','train','validation','--out_dir',$featureDir,'--batch_size','64')
    }
    Invoke-ReproductionStep '05_shared_checkpoint' @('-m','src.image.evaluate','--run_id','20261001_clip_vitb32_seed42','--features_dir',$featureDir,'--reference_metrics',$referenceMetrics,'--out_dir',"$evidenceDir/checkpoint_validation")
    Invoke-ReproductionStep '06_train' @('-m','src.image.train','--features_dir',$featureDir,'--run_id',$RunId)
    Invoke-ReproductionStep '07_verify_shared' @('-m','scripts.verify_predictions',"$evidenceDir/checkpoint_validation/predictions_validation.csv")
    Invoke-ReproductionStep '08_verify_training' @('-m','scripts.verify_predictions',"outputs/image/$RunId/predictions_validation.csv")
    Invoke-ReproductionStep '09_reload_checkpoint' @('-m','src.image.evaluate','--run_id',$RunId,'--features_dir',$featureDir)
    Invoke-ReproductionStep '10_compare_independent_training' @('-m','scripts.verify_reproduction','--reference_metrics',$referenceMetrics,'--candidate_metrics',"outputs/image/$RunId/metrics_validation.json")
    Invoke-ReproductionStep '11_compare_checkpoint_reload' @('-m','scripts.verify_reproduction','--reference_metrics',"outputs/image/$RunId/metrics_validation.json",'--candidate_metrics',"outputs/image/$RunId/reeval/metrics_validation.json",'--tolerance','0','--reference_predictions',"outputs/image/$RunId/predictions_validation.csv",'--candidate_predictions',"outputs/image/$RunId/reeval/predictions_validation.csv")
    Write-Output "Finished. Results: outputs/image/$RunId ; evidence: $evidenceDir"
}
finally { Pop-Location }
