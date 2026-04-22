# ears_lib

## Purpose

`ears_lib` is the reusable C++ library for local transcription pipelines.

Its job is to take audio chunks from a host, run them through library-owned pipeline stages, and emit transcription results back through a callback.

The library owns:

- stream lifecycle
- per-stream state
- VAD / ASR / optional LLM component selection
- scheduling and QoS
- stability / held-token behavior
- structured result and summary events
- config parsing and effective-config export

The host owns:

- audio capture
- text injection or UI output
- OS integration
- hotkeys and trigger UX
- app-specific storage and analytics

## Current Rewrite Status

This repo is mid-rewrite.

The important architectural decision is that the stream lifecycle API is the only canonical public surface:

1. create a stream
2. start the stream
3. push audio chunks
4. flush when needed
5. end the stream

Older one-shot thinking is not the center of the design anymore.

The rewrite is mainly about making this stream path the real path for library behavior, tests, and future host integrations.

## Core API

Main entry point: [include/ears/ears.hpp](../include/ears/ears.hpp)

Main supporting types: [include/ears/types.hpp](../include/ears/types.hpp)

Main config types: [include/ears/config.hpp](../include/ears/config.hpp)

Factory and backend registration API: [include/ears/factory.hpp](../include/ears/factory.hpp)

Canonical usage shape:

```cpp
#include "ears/ears.hpp"

ears::Config config;
ears::EarsPipeline pipeline(config);

pipeline.set_result_callback([](ears::TranscriptionResult const& result) {
  // handle partial/final results
});

ears::ContextProfile profile;
profile.profile_id = "default";

ears::StreamScheduling scheduling;
scheduling.qos_class = ears::StreamQosClass::latency_critical;

ears::StreamHandle stream =
    pipeline.create_stream(ears::StreamMode::realtime, profile, scheduling);

pipeline.start_stream(stream);
pipeline.push_audio(stream, chunk, ctx);
pipeline.flush_stream(stream);
pipeline.end_stream(stream);
```

## How The Library Works

At a high level, one chunk goes through this path:

1. `push_audio()` enqueues work for a stream.
2. The worker thread selects the next item based on flush barriers, QoS, priority hint, and queue order.
3. VAD decides whether the chunk looks like speech.
4. If speech is present, ASR runs.
5. If enabled, the LLM rectifier post-processes the ASR text.
6. Stability logic holds trailing words briefly before finalizing them.
7. The result callback receives `TranscriptionResult` events.

The main implementation for this is [src/pipeline.cpp](../src/pipeline.cpp).

## Main Parts Of The Codebase

- `include/ears/`: public headers and interfaces
- `src/config.cpp`: JSON parsing, defaults, and effective config export
- `src/factory.cpp`: registry and selection precedence
- `src/factory_backends.cpp`: backend registration wiring
- `src/pipeline.cpp`: stream state, worker loop, scheduling, and result emission
- `src/asr/`, `src/vad/`, `src/llm/`: concrete adapters and helpers
- `tests/`: library-only tests for config, factory behavior, pipeline behavior, and adapter helpers

## Config Model

The library is JSON-first.

Defaults come from [config/master.json](../config/master.json), with fallback defaults in `src/config.cpp` if the file is missing or invalid.

Important rules already implemented:

- unknown JSON keys are rejected
- invalid value ranges are rejected
- effective runtime config can be exported back to JSON
- effective config has a stable checksum

That makes config part of the library contract, not just startup convenience.

## Backend Model

The library treats VAD, ASR, and LLM as swappable components.

Implemented selection layers, from most specific to least specific, are:

1. provider-specific ASR factory
2. runtime-specific ASR factory
3. model-specific ASR factory
4. fallback dummy implementation

This selection logic lives in [src/factory.cpp](../src/factory.cpp).

Important reality check:

- the core abstraction is real
- some concrete backends are stubs or partial bridges
- tests often validate control flow and contract behavior, not full speech quality

That is expected in the current rewrite phase.

## Scheduling And Stream Rules

The stream API is asynchronous.

Implemented behavior to keep in mind:

- `push_audio()` returns enqueue status, not transcripts
- `flush_stream()` acts as a barrier and finalize point
- callbacks are ordered per stream
- heavy config changes are deferred until a flush boundary
- lightweight config changes apply on the next chunk
- `latency_critical` streams are protected ahead of lower-priority work
- thermal hints throttle lower-priority classes first

These rules are enforced in `src/pipeline.cpp` and covered heavily by [tests/pipeline_test.cpp](../tests/pipeline_test.cpp).

## What To Read First

If you are trying to understand or continue the rewrite, read in this order:

1. [include/ears/types.hpp](../include/ears/types.hpp)
2. [include/ears/ears.hpp](../include/ears/ears.hpp)
3. [tests/pipeline_test.cpp](../tests/pipeline_test.cpp)
4. [src/pipeline.cpp](../src/pipeline.cpp)
5. [include/ears/config.hpp](../include/ears/config.hpp)
6. [src/config.cpp](../src/config.cpp)
7. [include/ears/factory.hpp](../include/ears/factory.hpp)
8. [src/factory.cpp](../src/factory.cpp)

That path gets you the actual contract before the backend details.

## What Is In Scope Right Now

The practical focus for this library repo is:

- keep the stream API as the only real public path
- make config and factory behavior predictable
- keep library tests aligned with the stream contract
- make the library easy to consume from separate host repos
- continue separating real implemented behavior from future design ideas

## What Is Explicitly Out Of Scope Here

This repo should not grow host-app responsibilities back into itself.

Out of scope:

- app UX
- capture and recording utilities
- keyboard injection
- platform-specific host glue
- host-side benchmark wrappers and dashboards

Those belong in separate app repos that depend on `ears_lib`.

## Build Notes

Build entry points in this extracted repo are intentionally minimal:

```powershell
.\scripts\build_fast.ps1
.\scripts\build_all.ps1
```

The current scripts are lightweight wrappers around CMake build and test steps. They are meant to keep this repo small while the library rewrite settles.

## Final Take

Treat this repository as the library core under active simplification.

If you are unsure whether something belongs here, the rule of thumb is:

- if it is reusable and host-agnostic, it probably belongs in `ears_lib`
- if it touches capture, UI, OS glue, or app UX, it probably belongs in a separate host/app repo
