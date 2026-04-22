# ears_lib

ears_lib is the library-only split of Ears.

This repo is for the reusable cross-platform transcription library only. It intentionally excludes host apps, WAV utilities, recording helpers, and platform-specific app glue.

## Status

This library is in the middle of a rewrite.

The stable direction is:

- one canonical stream-based API
- config-driven backend selection
- library-owned scheduling, stability, and fail-soft behavior
- host apps living outside this repo and depending on the library separately

That means some code and adapter surface is more mature than others. The core pipeline, config, and factory structure matter most right now.

## What Is Here

- `include/`: public API
- `src/`: library implementation
- `config/master.json`: default runtime config
- `tests/`: library-owned tests
- `docs/ears_lib.md`: current library guide and rewrite notes

## What Is Not Here

- host applications
- OS-specific app integration
- WAV loading / recording helpers
- host integration tests
- dashboards and app-side benchmarking wrappers

## Build

```powershell
cmake -S . -B build
cmake --build build --config Release
ctest --test-dir build --build-config Release --output-on-failure
```

## Notes

- The CMake target is still named `ears` for now.
- ONNX Runtime is optional; without it, the project still builds in skeleton mode.
- Pre-1.0, the API may change as the rewrite settles.
