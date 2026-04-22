$ErrorActionPreference = 'Stop'

cmake -S . -B build
cmake --build build --config Release
ctest --test-dir build --build-config Release --output-on-failure
