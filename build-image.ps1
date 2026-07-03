param(
    [ValidateSet("build", "build-prod", "build-dev")]
    [string]$Command = "build-prod",
    [string]$ImageName = "echomind",
    [string]$Version = "latest",
    [switch]$NoCache,
    [string]$Platform = "",
    [string]$AptMirror = "https://mirrors.aliyun.com/debian",
    [string]$AptSecurityMirror = "https://mirrors.aliyun.com/debian-security",
    [string]$PipIndexUrl = "https://pypi.tuna.tsinghua.edu.cn/simple",
    [string]$PipTrustedHost = "pypi.tuna.tsinghua.edu.cn"
)

$ErrorActionPreference = "Stop"

$target = switch ($Command) {
    "build-prod" { "production" }
    "build-dev" { "development" }
    default { "" }
}

$imageWithVersion = "${ImageName}:${Version}"
$imageLatest = "${ImageName}:latest"

$args = @("build")

if ($target) {
    $args += @("--target", $target)
}

if ($NoCache) {
    $args += "--no-cache"
}

if ($Platform) {
    $args += @("--platform", $Platform)
}

$args += @(
    "--build-arg", "APT_MIRROR=$AptMirror",
    "--build-arg", "APT_SECURITY_MIRROR=$AptSecurityMirror",
    "--build-arg", "PIP_INDEX_URL=$PipIndexUrl",
    "--build-arg", "PIP_TRUSTED_HOST=$PipTrustedHost",
    "-t", $imageWithVersion,
    "-t", $imageLatest,
    "."
)

Write-Host "[INFO] Building image: $imageWithVersion"
Write-Host "[INFO] Command: docker $($args -join ' ')"
& docker @args

Write-Host "[INFO] Build complete: $imageWithVersion"
