# Maintainer: Yunis Torun
# Analysis period: 17-24 August 2026

$PythonExe = "C:\Users\Kapsam\.conda\envs\astro_scie\python.exe"
$OutDir = $PSScriptRoot

& $PythonExe -m pip freeze | Out-File -Encoding utf8 (Join-Path $OutDir "requirements.txt")
conda env export -n astro_scie --no-builds | Out-File -Encoding utf8 (Join-Path $OutDir "environment.yml")

Write-Host "Created environment\requirements.txt and environment\environment.yml from astro_scie."
