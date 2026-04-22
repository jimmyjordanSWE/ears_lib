param(
  [switch]$RunTests
)

$ErrorActionPreference = 'Stop'

cmake -S . -B build
cmake --build build --config Release

if ($RunTests) {
  ctest --test-dir build --build-config Release --output-on-failure
}
