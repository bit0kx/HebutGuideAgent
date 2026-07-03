param(
    [string]$ImageName = "echomind",
    [string]$Version = "latest",
    [string]$ContainerName = "echomind-app",
    [string]$EnvFile = ".env",
    [int]$ApiPort = 8000,
    [switch]$Detach,
    [string]$Restart = "unless-stopped"
)

$ErrorActionPreference = "Stop"

$imageTag = "${ImageName}:${Version}"

if (-not (Test-Path $EnvFile)) {
    throw "Env file not found: $EnvFile"
}

New-Item -ItemType Directory -Force -Path "data", "logs" | Out-Null

$dataPath = (Resolve-Path "data").Path
$logsPath = (Resolve-Path "logs").Path

$existing = docker ps -a --format "{{.Names}}" | Where-Object { $_ -eq $ContainerName }
if ($existing) {
    Write-Host "[INFO] Removing existing container: $ContainerName"
    docker stop $ContainerName 2>$null | Out-Null
    docker rm $ContainerName 2>$null | Out-Null
}

$args = @(
    "run",
    "--name", $ContainerName,
    "--restart", $Restart,
    "--env-file", $EnvFile,
    "-p", "${ApiPort}:8000",
    "-v", "${dataPath}:/app/data",
    "-v", "${logsPath}:/app/logs"
)

if ($Detach) {
    $args += "-d"
}

$args += $imageTag

Write-Host "[INFO] Starting container: $ContainerName"
Write-Host "[INFO] Image: $imageTag"
Write-Host "[INFO] Command: docker $($args -join ' ')"
& docker @args

Write-Host "[INFO] API: http://localhost:$ApiPort"
