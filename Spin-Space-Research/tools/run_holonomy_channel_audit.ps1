param(
    [int[]] $Seeds = @(1, 2, 6, 9),
    [string] $Python = 'python'
)

$ErrorActionPreference = 'Stop'

$repositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$harness = Join-Path $repositoryRoot 'src\mechanistic_group_actions.py'
$outputRoot = Join-Path $repositoryRoot 'artifacts'

foreach ($seed in $Seeds) {
    $stem = "mechanistic_a5_ga_holonomy_multiscale_channel_audit_seed{0}_1500" -f $seed
    Write-Output ("[{0:O}] starting seed {1}" -f (Get-Date), $seed)
    & $Python $harness `
        --device cuda `
        --group a5 `
        --input-elements 23145 31245 23451 51234 `
        --held-out-pairs 23145:23451 `
        --steps 1500 `
        --validation-batches 2 `
        --validation-batch-size 512 `
        --batch-size 256 `
        --diagnostic-interval 0 `
        --max-rotor-angle 2.2 `
        --holonomy-loss-weight 0.01 `
        --holonomy-loss-power 8 `
        --holonomy-margin-weight 0.1 `
        --holonomy-margin-target 0.5 `
        --holonomy-start-step 750 `
        --holonomy-ramp-steps 500 `
        --holonomy-word-multipliers 2 3 4 5 `
        --holonomy-batch-size 64 `
        --families pure_ga_rotor `
        --seed $seed `
        --output (Join-Path $outputRoot "$stem.json") `
        --checkpoint-output (Join-Path $outputRoot "$stem.pt")
    if ($LASTEXITCODE -ne 0) {
        throw "seed $seed failed with exit code $LASTEXITCODE"
    }
    Write-Output ("[{0:O}] completed seed {1}" -f (Get-Date), $seed)
}
