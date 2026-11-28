# Synthetic runner used only by low-memory-launcher.ps1. Never starts a game.
[pscustomobject]@{
    Arguments = $args
    Marker = $env:VICTORY_TEST_MARKER
    Profile = $env:VICTORY_MEMORY_PROFILE
    RayTracing = $env:VICTORY_BVH_BUILDER
    Dlss = $env:VICTORY_DLSS
}
$global:LASTEXITCODE = [int]$env:VICTORY_TEST_EXIT_CODE
