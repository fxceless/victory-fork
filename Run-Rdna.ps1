[CmdletBinding()]
param(
    # Hex GFX10 instruction words. The final result belongs in v0.
    [ValidateNotNullOrEmpty()][string]$Words = 'be861900 7e000206',
    # Four hex dwords are available in s0..s3 and v1..v4.
    [ValidateNotNullOrEmpty()][string]$Inputs = '80 0 0 0',
    [ValidateSet('NVIDIA', 'Intel')][string]$Adapter = 'NVIDIA'
)

$ErrorActionPreference = 'Stop'
if ($Words -notmatch '^(?:\s*(?:0x)?[a-f0-9]{1,8}\s+)*(?:0x)?[a-f0-9]{1,8}\s*$') {
    throw 'Words must contain space-separated hexadecimal dwords.'
}
if ($Inputs.Trim() -notmatch '^(?:(?:0x)?[a-f0-9]{1,8}\s+){3}(?:0x)?[a-f0-9]{1,8}$') {
    throw 'Inputs must contain four hexadecimal dwords.'
}
$runner = Join-Path $PSScriptRoot 'work/rdna-terminal.exe'
$names = 'RDNA_WORDS', 'RDNA_INPUTS', 'VICTORY_SHADER_ADAPTER', 'VICTORY_WARP'
$previous = foreach ($name in $names) { [pscustomobject]@{Name=$name; Value=[Environment]::GetEnvironmentVariable($name)} }
try {
    $env:RDNA_WORDS = $Words.ToLowerInvariant()
    $env:RDNA_INPUTS = $Inputs.ToLowerInvariant()
    $env:VICTORY_SHADER_ADAPTER = $Adapter
    $env:VICTORY_WARP = '0'
    & $runner --exact emit::valu_tests::terminal_rdna_compute --nocapture
    if ($LASTEXITCODE) { throw "RDNA runner exited with code $LASTEXITCODE." }
} finally {
    foreach ($entry in $previous) {
        if ($null -eq $entry.Value) {
            Remove-Item -LiteralPath ("Env:" + $entry.Name) -ErrorAction SilentlyContinue
        } else {
            [Environment]::SetEnvironmentVariable($entry.Name, $entry.Value)
        }
    }
}
