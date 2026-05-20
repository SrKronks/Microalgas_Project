param(
    [switch]$Full,
    [switch]$UpgradeTools
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

$CondaPython = "C:\Users\Asus\miniconda3\python.exe"
$PythonCommand = if (Test-Path $CondaPython) {
    @($CondaPython)
} elseif (Get-Command py -ErrorAction SilentlyContinue) {
    @("py", "-3.11")
} else {
    throw "No se encontro Python usable. Instala Miniconda/Python o agrega Python al PATH."
}

if (-not (Test-Path "venv\Scripts\python.exe")) {
    Write-Host "Creando entorno virtual en .\venv..."
    $PythonArgs = @()
    if ($PythonCommand.Count -gt 1) {
        $PythonArgs = $PythonCommand[1..($PythonCommand.Count - 1)]
    }
    & $PythonCommand[0] @PythonArgs -m venv venv
}

$Python = Join-Path $ProjectRoot "venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    throw "No se pudo crear .\venv\Scripts\python.exe"
}

if ($UpgradeTools) {
    Write-Host "Actualizando pip/setuptools/wheel..."
    & $Python -m pip install --disable-pip-version-check --upgrade pip setuptools wheel
}

$Requirements = if ($Full) { "requirements.txt" } else { "requirements-minimal.txt" }
Write-Host "Instalando dependencias desde $Requirements..."
& $Python -m pip install --disable-pip-version-check -r $Requirements

Write-Host ""
Write-Host "Entorno listo."
Write-Host "Activar con: .\venv\Scripts\activate"
Write-Host "Probar con:  python -m pytest -q"
