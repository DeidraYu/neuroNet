<#
.SYNOPSIS
    Configure the CMake build for Windows.

.DESCRIPTION
    Creates the build directory and generates the Visual Studio project files.
    Only needs to run once, or again after CMakeLists.txt changes.

    cmake is not on the PATH on a normal Visual Studio install, so this looks for it
    in the places it usually lives.

.EXAMPLE
    .\configure.ps1
    .\configure.ps1 -NoTuning     # portable binaries, no machine specific instructions
#>
param(
    [switch]$NoTuning
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot

. "$root\find-cmake.ps1"
$cmake = Find-CMake

$mnist = Join-Path (Split-Path $root -Parent) "mnist"
if (-not (Test-Path $mnist)) {
    Write-Host "The MNIST loader is missing. It has to sit next to this repository:" -ForegroundColor Yellow
    Write-Host "    cd $(Split-Path $root -Parent)" -ForegroundColor Yellow
    Write-Host "    git clone https://github.com/wichtounet/mnist.git" -ForegroundColor Yellow
    throw "MNIST not found at $mnist"
}

$arguments = @("-S", $root, "-B", (Join-Path $root "build"), "-G", "Visual Studio 17 2022", "-A", "x64")
if ($NoTuning) {
    $arguments += "-DNEURONET_TUNE_FOR_THIS_MACHINE=OFF"
}

Write-Host "Configuring..." -ForegroundColor Cyan
if ((Invoke-Native $cmake @arguments) -ne 0) { throw "configure failed" }

Write-Host "Configured. Next: .\build.ps1" -ForegroundColor Green
