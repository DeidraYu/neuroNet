<#
.SYNOPSIS
    Run the test suite.

.DESCRIPTION
    Builds first if that has not been done yet. Failing tests print their output.

.EXAMPLE
    .\test.ps1
    .\test.ps1 -Filter network
    .\test.ps1 -Config Debug
#>
param(
    [ValidateSet("Release", "Debug", "RelWithDebInfo")]
    [string]$Config = "Release",

    [string]$Filter = ""
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot

. "$root\find-cmake.ps1"
$ctest = Find-CTest

$buildDir = Join-Path $root "build"
if (-not (Test-Path (Join-Path $buildDir "CMakeCache.txt"))) {
    Write-Host "Not built yet, doing that first." -ForegroundColor Cyan
    & "$root\build.ps1" -Config $Config
}

$arguments = @("--test-dir", $buildDir, "-C", $Config, "--output-on-failure")
if ($Filter) {
    $arguments += @("-R", $Filter)
}

$code = Invoke-Native $ctest @arguments

if ($code -eq 0) {
    Write-Host "All tests passed." -ForegroundColor Green
} else {
    Write-Host "Tests failed." -ForegroundColor Red
}
exit $code
