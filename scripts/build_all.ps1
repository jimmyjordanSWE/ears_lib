$ErrorActionPreference = 'Stop'

$runtimeRoot = Join-Path $PSScriptRoot "..\..\ears\deps\onnxruntime-win-x64-gpu-1.24.2"
$runtimeRoot = [System.IO.Path]::GetFullPath($runtimeRoot)
if (Test-Path $runtimeRoot) {
  $env:ONNXRUNTIME_ROOT = $runtimeRoot
  $env:PATH = (Join-Path $runtimeRoot "lib") + ";" + $env:PATH
  $cmakeArgs = @("-S", ".", "-B", "build", "-DONNXRUNTIME_ROOT=$runtimeRoot", "-DEARS_USE_CUDA=ON")
} else {
  $cmakeArgs = @("-S", ".", "-B", "build")
}

cmake @cmakeArgs
cmake --build build --config Release

if (Test-Path $runtimeRoot) {
  $outDir = Join-Path (Join-Path (Get-Location) "build") "Release"
  if (Test-Path $outDir) {
    Copy-Item (Join-Path $runtimeRoot "lib\\onnxruntime.dll") $outDir -Force
    Copy-Item (Join-Path $runtimeRoot "lib\\onnxruntime_providers_shared.dll") $outDir -Force
    Copy-Item (Join-Path $runtimeRoot "lib\\onnxruntime_providers_cuda.dll") $outDir -Force
  }
}

ctest --test-dir build --build-config Release --output-on-failure
