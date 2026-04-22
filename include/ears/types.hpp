#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace ears {

enum class StreamMode {
  realtime,
  batch,
};

using StreamHandle = uint64_t;

enum class StreamQosClass {
  latency_critical,
  latency_sensitive,
  thermal_constrained,
  opportunistic_idle,
};

enum class TriggerMode {
  tap_to_start_vad_to_end,
  hold_to_talk,
  always_on_vad,
};

struct StreamScheduling {
  StreamQosClass qos_class = StreamQosClass::latency_critical;
  TriggerMode trigger_mode = TriggerMode::tap_to_start_vad_to_end;
  float priority_hint = 0.5f;
};

enum class EnqueueStatus {
  ok,
  backpressure,
  invalid_state,
};

enum class ThermalHint {
  normal,
  reduce_gpu,
  minimum_gpu,
};

struct ContextProfile {
  std::string profile_id = "default";
  std::string cleanup_prompt;
  std::string locale;
  std::string domain;
  std::unordered_map<std::string, std::string> hints;
  int version = 1;
};

struct PipelineProfileSpec {
  std::string pipeline_id;
  StreamMode mode = StreamMode::realtime;
  ContextProfile profile;
  StreamScheduling scheduling;
};

struct AudioChunk {
  std::vector<float> samples;  // mono, 16 kHz (or configurable)
  int64_t timestamp_ns = 0;    // host-provided, high-res clock
  std::string tier = "tier1";  // "tier1" (realtime) | "tier2" (batch)
};

struct Context {
  std::string app_window_title;  // e.g., "Visual Studio Code"
  std::string process_name;      // optional
};

struct TranscriptionResult {
  StreamHandle stream_id = 0;
  uint64_t sequence_no = 0;
  std::string corrected_text;  // final text to emit (after stability)
  std::string raw_asr_text;    // phonetic output from ASR
  std::string raw_asr_json;    // full ASR output for logging
  int64_t timestamp_ns = 0;
  int64_t start_ms = 0;
  int64_t end_ms = 0;
  bool is_final = false;
  std::string mode = "realtime";
  std::string status = "ok";
  std::string error_message;
  std::string profile_id;
  int profile_version = 0;
  float confidence = 0.f;
  std::string model_id;
  std::string provider;
  float decode_ms = 0.f;
  float queue_delay_ms = 0.f;
  float end_to_end_ms = 0.f;
};

struct StreamSummary {
  StreamHandle stream_id = 0;
  uint64_t chunks_seen = 0;
  uint64_t results_emitted = 0;
  uint64_t errors = 0;
  float p50_end_to_end_ms = 0.f;
  float p95_end_to_end_ms = 0.f;
  float p99_end_to_end_ms = 0.f;
  float avg_end_to_end_ms = 0.f;
  float peak_end_to_end_ms = 0.f;
};

}  // namespace ears
