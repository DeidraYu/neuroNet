<#
.SYNOPSIS
    Build the project on Windows.

.DESCRIPTION
    Configures first if that has not been done yet.

.EXAMPLE
    .\build.ps1
    .\build.ps1 -Config Debug
#>
param(
    [ValidateSet("Release", "Debug", "RelWithDebInfo")]
    [string]$Config = "Release"
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot

. "$root\find-cmake.ps1"
$cmake = Find-CMake

$buildDir = Join-Path $root "build"
if (-not (Test-Path (Join-Path $buildDir "CMakeCache.txt"))) {
    Write-Host "Not configured yet, doing that first." -ForegroundColor Cyan
    & "$root\configure.ps1"
}

Write-Host "Building ($Config)..." -ForegroundColor Cyan
if ((Invoke-Native $cmake --build $buildDir --config $Config) -ne 0) { throw "build failed" }

Write-Host "Built. Next: .\run.ps1 or .\test.ps1" -ForegroundColor Green
