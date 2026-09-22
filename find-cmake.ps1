<#
.SYNOPSIS
    Locate cmake and ctest on a Windows machine.

.DESCRIPTION
    Dot-sourced by the other scripts. A Visual Studio install ships cmake but does not
    put it on the PATH, so this checks the PATH first and then the usual install
    locations.
#>

function Find-CMakeDirectory {
    $onPath = Get-Command cmake -ErrorAction SilentlyContinue
    if ($onPath) {
        return (Split-Path $onPath.Source -Parent)
    }

    $editions = @("Community", "Professional", "Enterprise", "BuildTools")
    $years = @("2022", "2019")
    $roots = @("C:\Program Files\Microsoft Visual Studio", "C:\Program Files (x86)\Microsoft Visual Studio")

    foreach ($root in $roots) {
        foreach ($year in $years) {
            foreach ($edition in $editions) {
                $candidate = Join-Path $root "$year\$edition\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin"
                if (Test-Path (Join-Path $candidate "cmake.exe")) {
                    return $candidate
                }
            }
        }
    }

    if (Test-Path "C:\Program Files\CMake\bin\cmake.exe") {
        return "C:\Program Files\CMake\bin"
    }

    throw "cmake was not found. Install CMake, or install the C++ workload in Visual Studio."
}

function Find-CMake {
    return (Join-Path (Find-CMakeDirectory) "cmake.exe")
}

function Find-CTest {
    return (Join-Path (Find-CMakeDirectory) "ctest.exe")
}

<#
.SYNOPSIS
    Run an external program and fail only on a non zero exit code.

.DESCRIPTION
    With $ErrorActionPreference set to Stop, PowerShell treats anything a program writes
    to stderr as a terminating error. cmake writes its warnings there, so a plain call
    would abort on a warning. This merges the two streams and judges success by the exit
    code, which is what the program actually meant.
#>
function Invoke-Native {
    param(
        [Parameter(Mandatory = $true)][string]$Executable,
        [Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments
    )

    $previous = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        # Written to the host rather than returned, so the only thing this function
        # produces is the exit code. Lines that arrived on stderr are error records
        # rather than strings, so they are converted rather than printed as objects.
        & $Executable @Arguments 2>&1 | ForEach-Object { Write-Host $_.ToString() }
        return $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previous
    }
}
