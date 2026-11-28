# Exercise the production launcher's settings without a binary, game or GPU.
$ErrorActionPreference = 'Stop'
$launcher = Join-Path $PSScriptRoot '..\scripts\play-low-memory.ps1'
$checked = 0
foreach ($mode in @('stock', 'quality', 'balanced', 'performance')) {
    $plan = & $launcher -Exe 'unused.exe' -Upscaling $mode -ShowSettings
    if ($plan.Console -ne 'series-s') { throw 'The memory profile must use Series S.' }
    if ($plan.Settings.VICTORY_MEMORY_PROFILE -ne '16gb-8gb') { throw 'Small-memory policy is missing.' }
    if ($plan.Settings.VICTORY_MEMORY_PROFILE -ne '16gb-8gb' -or $plan.Settings.VICTORY_TEXTURE_BUDGET_MB -ne '2560') { throw 'Small-memory profile and supplemental texture cap must stay enabled.' }
    if ($plan.Settings.ContainsKey('VICTORY_POOL_GATE')) { throw 'Removed upstream block-gate switch must not be advertised.' }
    if ($plan.Settings.VICTORY_DLSS -ne '0' -or $plan.Settings.VICTORY_FRAMEGEN -ne '0') { throw 'Default launch must not request NVIDIA-only effects.' }
    if ($plan.Settings.ContainsKey('VICTORY_D3D12_ADAPTER')) { throw 'The profile must not force an NVIDIA adapter.' }
    if ($plan.Settings.VICTORY_BVH_BUILDER -ne '0') { throw 'Ray tracing must be opt-in on the small-memory profile.' }
    $expectedFsr = if ($mode -eq 'stock') { '0' } else { '1' }
    if ($plan.Settings.VICTORY_FSR_PATH -ne $expectedFsr) { throw "Wrong FSR setting for $mode." }
    if ($mode -ne 'stock' -and $plan.Settings.VICTORY_DLSS_MODE -ne $mode) { throw 'The native upscaler mode was not forwarded.' }
    $checked++
}
$plan = & $launcher -Exe 'unused.exe' -RayTracing -More @{VICTORY_TEXTURE_BUDGET_MB='2048'} -ShowSettings
if ($plan.Settings.VICTORY_BVH_BUILDER -ne '1' -or $plan.Settings.VICTORY_TEXTURE_BUDGET_MB -ne '2048') { throw 'Explicit overrides were not preserved.' }
$probe = Join-Path $PSScriptRoot 'launcher-probe.ps1'
$inputFile = Join-Path $PSScriptRoot '..\README.md'
$previousMarker = $env:VICTORY_TEST_MARKER
$previousProfile = $env:VICTORY_MEMORY_PROFILE
$originalLocation = (Get-Location).Path
try {
    $env:VICTORY_TEST_MARKER = 'parent-value'
    $env:VICTORY_MEMORY_PROFILE = 'parent-profile'
    $observed = & $launcher -Exe $inputFile -Runner $probe -More @{ VICTORY_TEST_MARKER='child-value'; VICTORY_TEST_EXIT_CODE='0' }
    if ($observed.Marker -ne 'child-value' -or $observed.Profile -ne '16gb-8gb' -or $observed.RayTracing -ne '0' -or $observed.Dlss -ne '0') { throw 'The actual invocation did not receive its settings.' }
    if (($observed.Arguments -join '|') -notmatch '\|--console\|series-s\|--execute$') { throw 'Actual invocation lost the console argument.' }
    if ($env:VICTORY_TEST_MARKER -ne 'parent-value' -or $env:VICTORY_MEMORY_PROFILE -ne 'parent-profile') { throw 'Successful launch polluted parent settings.' }
    try {
        & $launcher -Exe $inputFile -Runner $probe -More @{ VICTORY_TEST_EXIT_CODE='7' } | Out-Null
        throw 'A failing runner was reported as success.'
    } catch {
        if ($_.Exception.Message -notmatch 'exited with code 7') { throw }
    }
    if ($env:VICTORY_TEST_MARKER -ne 'parent-value' -or $env:VICTORY_MEMORY_PROFILE -ne 'parent-profile') { throw 'Failed launch polluted parent settings.' }
    if ((Get-Location).Path -ne $originalLocation) { throw 'Launch failed to restore the working directory.' }
    try {
        & $launcher -Exe $inputFile -Runner (Join-Path $PSScriptRoot 'missing-runner.exe') | Out-Null
        throw 'Missing runner was accepted.'
    } catch {
        if ($_.Exception.Message -notmatch 'Build it first') { throw }
    }
} finally {
    $env:VICTORY_TEST_MARKER = $previousMarker
    $env:VICTORY_MEMORY_PROFILE = $previousProfile
    $global:LASTEXITCODE = 0
}
$tokens = $null; $parseErrors = $null
[System.Management.Automation.Language.Parser]::ParseFile((Join-Path $PSScriptRoot '..\scripts\play.ps1'), [ref]$tokens, [ref]$parseErrors) | Out-Null
if ($parseErrors.Count) { throw 'Generic launcher syntax errors.' }
$generic = Get-Content (Join-Path $PSScriptRoot '..\scripts\play.ps1') -Raw
if ($generic -notmatch '--console \$Console') { throw 'Generic launcher does not forward console selection.' }
Write-Output "PASS: $checked upscaler modes, overrides, actual console/settings forwarding, success/failure environment restoration, missing runner rejection."
