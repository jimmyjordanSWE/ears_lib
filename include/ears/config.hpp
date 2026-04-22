#pragma once

#include <string>
#include <unordered_map>

namespace ears {

struct VadConfig {
  VadConfig();
  std::string model;
  std::string path;
  float threshold = 0.0f;
  int hangover_ms = 0;
};

struct AsrConfig {
  AsrConfig();
  std::string model;
  std::string family;
  std::string runtime;
  std::string path;
  int chunk_ms = 0;
  int beam_size = 0;
};

struct LlmConfig {
  LlmConfig();
  bool enabled = false;
  std::string model;
  std::string path;
  std::string quantization;
};

struct StabilityConfig {
  StabilityConfig();
  int hold_ms = 0;
  int history_ms = 0;
  int hold_words = 0;
};

struct Config {
  Config();
  VadConfig vad;
  AsrConfig asr;
  LlmConfig llm;
  StabilityConfig stability;
  std::string provider;  // cpu | cuda | coreml | directml | tensorrt
};

// Load config from JSON file. Throws on error.
Config load_config(std::string const& path);

// Load config from JSON string. Throws on error.
Config load_config_from_string(std::string const& json);

// Import config from JSON payload. Throws on error.
// This is the canonical JSON bootstrap path for startup from serialized config.
Config import_config_json(std::string const& json);

// Export fully-resolved runtime config to JSON.
// The output contains all current fields and values from Config.
std::string export_effective_config_json(Config const& config);

// Stable checksum for the effective runtime config JSON payload.
std::string effective_config_checksum_hex(Config const& config);

}  // namespace ears
