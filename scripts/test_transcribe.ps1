param(
  [string]$InputFile = "test_audio.flac",
  [string]$Provider = "cuda",
  [string]$ModelDir = "models/whisper-medium_timestamped/onnx"
)

$ErrorActionPreference = 'Stop'

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$inputPath = [System.IO.Path]::GetFullPath((Join-Path $repoRoot $InputFile))
$modelPath = [System.IO.Path]::GetFullPath((Join-Path $repoRoot $ModelDir))
$rawPath = Join-Path $repoRoot "build\test_audio.f32le"
$exePath = Join-Path $repoRoot "build\Release\ears_transcribe_cli.exe"

if (!(Test-Path $inputPath)) {
  throw "Input file not found: $inputPath"
}

ffmpeg -y -i $inputPath -ac 1 -ar 16000 -f f32le $rawPath

if (!(Test-Path $exePath)) {
  . (Join-Path $PSScriptRoot "build_fast.ps1")
}

& $exePath $modelPath $rawPath $Provider
