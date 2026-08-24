# Maintainer: Yunis Torun
# Analysis period: 17-24 August 2026

$PythonExe = "C:\Users\Kapsam\.conda\envs\astro_scie\python.exe"

& $PythonExe -m pip freeze | Out-File -Encoding utf8 requirements.txt
conda env export -n astro_scie --no-builds | Out-File -Encoding utf8 environment.yml

Write-Host "Created requirements.txt and environment.yml from astro_scie."
