[CmdletBinding()]
param([switch]$RequireRust)
$ErrorActionPreference = 'Stop'
$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$errorsFound = @()
foreach ($file in @('scripts\play.ps1', 'scripts\play-low-memory.ps1', 'tests\low-memory-launcher.ps1', 'tests\launcher-probe.ps1', 'scripts\check-local-fixes.ps1')) {
    $tokens = $null; $parseErrors = $null
    [System.Management.Automation.Language.Parser]::ParseFile((Join-Path $root $file), [ref]$tokens, [ref]$parseErrors) | Out-Null
    $errorsFound += $parseErrors
}
if ($errorsFound.Count) { throw ($errorsFound | Out-String) }
& (Join-Path $root 'tests\low-memory-launcher.ps1')
$cargo = Get-Command cargo -ErrorAction SilentlyContinue
if (-not $cargo) {
    if ($RequireRust) { throw 'Rust/Cargo is missing: Rust build, Clippy, unit tests and gameplay remain unverified.' }
    Write-Warning 'Launcher checks passed. Rust/Cargo is missing; Rust integration and gameplay remain unverified.'
    return
}
Push-Location $root
try {
    & $cargo.Source clippy --workspace --all-targets --locked -- -D warnings
    if ($LASTEXITCODE -ne 0) { throw 'Workspace Clippy failed.' }
    & $cargo.Source clippy --workspace --all-targets --all-features --locked -- -D warnings
    if ($LASTEXITCODE -ne 0) { throw 'Diagnostics Clippy failed.' }
    & $cargo.Source test --workspace --locked
    if ($LASTEXITCODE -ne 0) { throw 'Workspace tests failed.' }
    & $cargo.Source build --release --locked -p victory
    if ($LASTEXITCODE -ne 0) { throw 'Release build failed.' }
} finally { Pop-Location }
