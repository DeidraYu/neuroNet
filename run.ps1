<#
.SYNOPSIS
    Train the network.

.DESCRIPTION
    Builds first if the executable is missing, then runs it. Anything after the
    options below is handed straight to the program:

        <epochs> <miniBatchSize> <learningRate> "(layer,sizes)"

    With no arguments it uses the defaults in main.cpp.

.EXAMPLE
    .\run.ps1
    .\run.ps1 1 10 3.0 "(784,30,10)"
    .\run.ps1 -Config Debug 1 10 3.0 "(784,30,10)"
#>
param(
    [ValidateSet("Release", "Debug", "RelWithDebInfo")]
    [string]$Config = "Release",

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Arguments
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot

$exe = Join-Path $root "build\src\$Config\neuronet.exe"
if (-not (Test-Path $exe)) {
    Write-Host "Not built yet, doing that first." -ForegroundColor Cyan
    & "$root\build.ps1" -Config $Config
}

if ($Arguments) {
    & $exe @Arguments
} else {
    & $exe
}
exit $LASTEXITCODE
