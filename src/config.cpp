#include "ears/config.hpp"

#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <nlohmann/json.hpp>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>

#include "ears/internal/env_utils.hpp"
#include "ears/internal/runtime_id_utils.hpp"
#include "ears/internal/string_utils.hpp"

namespace ears {

namespace {

constexpr int kConfigSchemaVersion = 1;

struct DefaultVadValues {
  std::string model;
  std::string path;
  float threshold = 0.0f;
  int hangover_ms = 0;
};

struct DefaultAsrValues {
  std::string model;
  std::string family;
  std::string runtime;
  std::string path;
  int chunk_ms = 0;
  int beam_size = 0;
};

struct DefaultLlmValues {
  bool enabled = false;
  std::string model;
  std::string path;
  std::string quantization;
};

struct DefaultStabilityValues {
  int hold_ms = 0;
  int history_ms = 0;
  int hold_words = 0;
};

struct DefaultConfigValues {
  DefaultVadValues vad;
  DefaultAsrValues asr;
  DefaultLlmValues llm;
  DefaultStabilityValues stability;
  std::string provider;
};

DefaultConfigValues builtin_default_values() {
  DefaultConfigValues defaults;
  defaults.vad.model = "silero";
  defaults.vad.path = "models/silero_v5.onnx";
  defaults.vad.threshold = 0.8f;
  defaults.vad.hangover_ms = 400;

  defaults.asr.model = "whisper";
  defaults.asr.family = "whisper_seq2seq";
  defaults.asr.runtime = "onnx";
  defaults.asr.path = "models/whisper-medium_timestamped/onnx";
  defaults.asr.chunk_ms = 500;
  defaults.asr.beam_size = 1;

  defaults.llm.enabled = false;
  defaults.llm.model = "smollm2";
  defaults.llm.path = "models/smollm2_360m.onnx";
  defaults.llm.quantization = "int4";

  defaults.stability.hold_ms = 250;
  defaults.stability.history_ms = 1000;
  defaults.stability.hold_words = 3;

  defaults.provider = "auto";
  return defaults;
}

std::string master_config_path() {
  std::string runtime_path = internal::getenv_copy("EARS_MASTER_CONFIG_PATH");
  if (!runtime_path.empty()) {
    return runtime_path;
  }
#ifdef EARS_MASTER_CONFIG_PATH
  return EARS_MASTER_CONFIG_PATH;
#else
  return "config/master.json";
#endif
}

bool strict_startup_mode_enabled() {
  std::string value = internal::getenv_copy("EARS_STRICT_STARTUP");
  if (value.empty()) {
    return false;
  }

  std::string normalized = internal::to_lower_ascii(value);
  return normalized == "1" || normalized == "true" || normalized == "yes" || normalized == "on";
}

bool read_file(std::string const& path, std::string& out) {
  std::ifstream in(path);
  if (!in.is_open()) {
    return false;
  }
  std::stringstream ss;
  ss << in.rdbuf();
  out = ss.str();
  return true;
}

void validate_object_keys(nlohmann::json const& object,
                          std::initializer_list<char const*> allowed_keys,
                          std::string const& section_name) {
  if (!object.is_object()) {
    throw std::runtime_error("Config section '" + section_name + "' must be an object");
  }

  std::unordered_set<std::string> allowed;
  allowed.reserve(allowed_keys.size());
  for (char const* key : allowed_keys) {
    allowed.emplace(key);
  }

  for (auto const& entry : object.items()) {
    if (allowed.find(entry.key()) == allowed.end()) {
      throw std::runtime_error("Unknown config key: '" + section_name + "." + entry.key() + "'");
    }
  }
}

void validate_schema_version(nlohmann::json const& root, bool require_field) {
  if (!root.contains("schema_version")) {
    if (require_field) {
      throw std::runtime_error("Missing required config key: 'schema_version'");
    }
    return;
  }

  auto const& version_node = root["schema_version"];
  if (!version_node.is_number_integer()) {
    throw std::runtime_error("Config key 'schema_version' must be an integer");
  }

  int const schema_version = version_node.get<int>();
  if (schema_version != kConfigSchemaVersion) {
    throw std::runtime_error(
        "Unsupported config schema_version: " + std::to_string(schema_version) + " (expected " +
        std::to_string(kConfigSchemaVersion) + ")");
  }
}

VadConfig parse_vad(nlohmann::json const& j);
AsrConfig parse_asr(nlohmann::json const& j);
LlmConfig parse_llm(nlohmann::json const& j);
StabilityConfig parse_stability(nlohmann::json const& j);

Config parse_config_json_object(nlohmann::json const& j) {
  validate_object_keys(j, {"schema_version", "vad", "asr", "llm", "stability", "provider"}, "root");
  Config c;
  c.vad = parse_vad(j);
  c.asr = parse_asr(j);
  c.llm = parse_llm(j);
  c.stability = parse_stability(j);
  if (j.contains("provider")) {
    c.provider = j["provider"].get<std::string>();
  }
  auto validate_numeric_values = [](float vad_threshold, int vad_hangover_ms, int asr_chunk_ms,
                                    int asr_beam_size, int stability_hold_ms,
                                    int stability_history_ms, int stability_hold_words) {
    if (!std::isfinite(vad_threshold) || vad_threshold < 0.0f || vad_threshold > 1.0f) {
      throw std::runtime_error("Invalid config value: vad.threshold must be in [0.0, 1.0]");
    }
    if (vad_hangover_ms < 0) {
      throw std::runtime_error("Invalid config value: vad.hangover_ms must be >= 0");
    }
    if (asr_chunk_ms <= 0) {
      throw std::runtime_error("Invalid config value: asr.chunk_ms must be > 0");
    }
    if (asr_beam_size <= 0) {
      throw std::runtime_error("Invalid config value: asr.beam_size must be > 0");
    }
    if (stability_hold_ms < 0) {
      throw std::runtime_error("Invalid config value: stability.hold_ms must be >= 0");
    }
    if (stability_history_ms < 0) {
      throw std::runtime_error("Invalid config value: stability.history_ms must be >= 0");
    }
    if (stability_hold_words < 0) {
      throw std::runtime_error("Invalid config value: stability.hold_words must be >= 0");
    }
  };
  auto validate_runtime_provider_combo = [](std::string const& runtime_value,
                                            std::string const& provider_value) {
    std::string const runtime = internal::normalize_runtime_id(runtime_value);
    std::string const provider = internal::normalize_provider_id(provider_value);

    if (provider.empty() || provider == "auto") {
      return;
    }

    auto invalid = [&]() {
      throw std::runtime_error("Malformed runtime/provider combination: runtime='" + runtime +
                               "', provider='" + provider + "'");
    };

    if (runtime == "onnx_runtime") {
      static std::unordered_set<std::string> const allowed = {"cuda", "directml"};
      if (allowed.find(provider) == allowed.end()) {
        invalid();
      }
      return;
    }
    if (runtime == "tensorrt") {
      if (provider != "tensorrt" && provider != "cuda" && provider != "cpu") {
        invalid();
      }
      return;
    }
    if (runtime == "openvino") {
      if (provider != "openvino_device" && provider != "cpu") {
        invalid();
      }
      return;
    }
    if (runtime == "coreml") {
      if (provider != "coreml_ep" && provider != "cpu") {
        invalid();
      }
      return;
    }
    if (runtime == "qnn") {
      if (provider != "qnn_ep" && provider != "cpu") {
        invalid();
      }
      return;
    }
  };

  validate_numeric_values(c.vad.threshold, c.vad.hangover_ms, c.asr.chunk_ms, c.asr.beam_size,
                          c.stability.hold_ms, c.stability.history_ms, c.stability.hold_words);
  validate_runtime_provider_combo(c.asr.runtime, c.provider);
  return c;
}

void apply_defaults_from_json(nlohmann::json const& j, DefaultConfigValues& defaults) {
  validate_object_keys(j, {"schema_version", "vad", "asr", "llm", "stability", "provider"}, "root");
  validate_schema_version(j, false);

  if (j.contains("vad")) {
    auto const& v = j["vad"];
    validate_object_keys(v, {"model", "path", "threshold", "hangover_ms"}, "vad");
    if (v.contains("model"))
      defaults.vad.model = v["model"].get<std::string>();
    if (v.contains("path"))
      defaults.vad.path = v["path"].get<std::string>();
    if (v.contains("threshold"))
      defaults.vad.threshold = v["threshold"].get<float>();
    if (v.contains("hangover_ms"))
      defaults.vad.hangover_ms = v["hangover_ms"].get<int>();
  }

  if (j.contains("asr")) {
    auto const& a = j["asr"];
    validate_object_keys(a, {"model", "family", "runtime", "path", "chunk_ms", "beam_size"}, "asr");
    if (a.contains("model"))
      defaults.asr.model = a["model"].get<std::string>();
    if (a.contains("family"))
      defaults.asr.family = a["family"].get<std::string>();
    if (a.contains("runtime"))
      defaults.asr.runtime = a["runtime"].get<std::string>();
    if (a.contains("path"))
      defaults.asr.path = a["path"].get<std::string>();
    if (a.contains("chunk_ms"))
      defaults.asr.chunk_ms = a["chunk_ms"].get<int>();
    if (a.contains("beam_size"))
      defaults.asr.beam_size = a["beam_size"].get<int>();
  }

  if (j.contains("llm")) {
    auto const& l = j["llm"];
    validate_object_keys(l, {"enabled", "model", "path", "quantization"}, "llm");
    if (l.contains("enabled"))
      defaults.llm.enabled = l["enabled"].get<bool>();
    if (l.contains("model"))
      defaults.llm.model = l["model"].get<std::string>();
    if (l.contains("path"))
      defaults.llm.path = l["path"].get<std::string>();
    if (l.contains("quantization"))
      defaults.llm.quantization = l["quantization"].get<std::string>();
  }

  if (j.contains("stability")) {
    auto const& s = j["stability"];
    validate_object_keys(s, {"hold_ms", "history_ms", "hold_words"}, "stability");
    if (s.contains("hold_ms"))
      defaults.stability.hold_ms = s["hold_ms"].get<int>();
    if (s.contains("history_ms"))
      defaults.stability.history_ms = s["history_ms"].get<int>();
    if (s.contains("hold_words"))
      defaults.stability.hold_words = s["hold_words"].get<int>();
  }

  if (j.contains("provider")) {
    defaults.provider = j["provider"].get<std::string>();
  }

  if (!std::isfinite(defaults.vad.threshold) || defaults.vad.threshold < 0.0f ||
      defaults.vad.threshold > 1.0f) {
    throw std::runtime_error("Invalid config value: vad.threshold must be in [0.0, 1.0]");
  }
  if (defaults.vad.hangover_ms < 0) {
    throw std::runtime_error("Invalid config value: vad.hangover_ms must be >= 0");
  }
  if (defaults.asr.chunk_ms <= 0) {
    throw std::runtime_error("Invalid config value: asr.chunk_ms must be > 0");
  }
  if (defaults.asr.beam_size <= 0) {
    throw std::runtime_error("Invalid config value: asr.beam_size must be > 0");
  }
  if (defaults.stability.hold_ms < 0) {
    throw std::runtime_error("Invalid config value: stability.hold_ms must be >= 0");
  }
  if (defaults.stability.history_ms < 0) {
    throw std::runtime_error("Invalid config value: stability.history_ms must be >= 0");
  }
  if (defaults.stability.hold_words < 0) {
    throw std::runtime_error("Invalid config value: stability.hold_words must be >= 0");
  }

  std::string const normalized_runtime = internal::normalize_runtime_id(defaults.asr.runtime);
  std::string const normalized_provider = internal::normalize_provider_id(defaults.provider);
  if (!normalized_provider.empty() && normalized_provider != "auto") {
    bool valid_combo = true;
    if (normalized_runtime == "onnx_runtime") {
      static std::unordered_set<std::string> const allowed = {"cuda", "directml"};
      valid_combo = (allowed.find(normalized_provider) != allowed.end());
    } else if (normalized_runtime == "tensorrt") {
      valid_combo = (normalized_provider == "tensorrt" || normalized_provider == "cuda" ||
                     normalized_provider == "cpu");
    } else if (normalized_runtime == "openvino") {
      valid_combo = (normalized_provider == "openvino_device" || normalized_provider == "cpu");
    } else if (normalized_runtime == "coreml") {
      valid_combo = (normalized_provider == "coreml_ep" || normalized_provider == "cpu");
    } else if (normalized_runtime == "qnn") {
      valid_combo = (normalized_provider == "qnn_ep" || normalized_provider == "cpu");
    }
    if (!valid_combo) {
      throw std::runtime_error("Malformed runtime/provider combination: runtime='" +
                               normalized_runtime + "', provider='" + normalized_provider + "'");
    }
  }
}

DefaultConfigValues load_master_default_values() {
  DefaultConfigValues defaults = builtin_default_values();
  std::string json_text;
  std::string const path = master_config_path();
  if (!read_file(path, json_text)) {
    if (strict_startup_mode_enabled()) {
      throw std::runtime_error("Strict startup: missing master config file: " + path);
    }
    return defaults;
  }

  try {
    nlohmann::json parsed = nlohmann::json::parse(json_text);
    apply_defaults_from_json(parsed, defaults);
  } catch (std::exception const& e) {
    if (strict_startup_mode_enabled()) {
      throw std::runtime_error("Strict startup: invalid master config '" + path + "': " + e.what());
    }
    // Keep builtin defaults if master JSON is malformed.
    return defaults;
  }
  return defaults;
}

DefaultConfigValues const& default_values() {
  static std::mutex defaults_mu;
  static bool initialized = false;
  static std::string cached_master_path;
  static bool cached_strict_mode = false;
  static DefaultConfigValues defaults;

  std::string const current_master_path = master_config_path();
  bool const current_strict_mode = strict_startup_mode_enabled();

  std::lock_guard<std::mutex> lock(defaults_mu);
  if (!initialized || current_master_path != cached_master_path ||
      current_strict_mode != cached_strict_mode) {
    defaults = load_master_default_values();
    cached_master_path = current_master_path;
    cached_strict_mode = current_strict_mode;
    initialized = true;
  }
  return defaults;
}

VadConfig parse_vad(nlohmann::json const& j) {
  VadConfig c;
  if (j.contains("vad")) {
    auto const& v = j["vad"];
    validate_object_keys(v, {"model", "path", "threshold", "hangover_ms"}, "vad");
    if (v.contains("model"))
      c.model = v["model"].get<std::string>();
    if (v.contains("path"))
      c.path = v["path"].get<std::string>();
    if (v.contains("threshold"))
      c.threshold = v["threshold"].get<float>();
    if (v.contains("hangover_ms"))
      c.hangover_ms = v["hangover_ms"].get<int>();
  }
  return c;
}

AsrConfig parse_asr(nlohmann::json const& j) {
  AsrConfig c;
  if (j.contains("asr")) {
    auto const& a = j["asr"];
    validate_object_keys(a, {"model", "family", "runtime", "path", "chunk_ms", "beam_size"}, "asr");
    if (a.contains("model"))
      c.model = a["model"].get<std::string>();
    if (a.contains("family"))
      c.family = a["family"].get<std::string>();
    if (a.contains("runtime"))
      c.runtime = a["runtime"].get<std::string>();
    if (a.contains("path"))
      c.path = a["path"].get<std::string>();
    if (a.contains("chunk_ms"))
      c.chunk_ms = a["chunk_ms"].get<int>();
    if (a.contains("beam_size"))
      c.beam_size = a["beam_size"].get<int>();
  }
  return c;
}

LlmConfig parse_llm(nlohmann::json const& j) {
  LlmConfig c;
  if (j.contains("llm")) {
    auto const& l = j["llm"];
    validate_object_keys(l, {"enabled", "model", "path", "quantization"}, "llm");
    if (l.contains("enabled"))
      c.enabled = l["enabled"].get<bool>();
    if (l.contains("model"))
      c.model = l["model"].get<std::string>();
    if (l.contains("path"))
      c.path = l["path"].get<std::string>();
    if (l.contains("quantization"))
      c.quantization = l["quantization"].get<std::string>();
  }
  return c;
}

StabilityConfig parse_stability(nlohmann::json const& j) {
  StabilityConfig c;
  if (j.contains("stability")) {
    auto const& s = j["stability"];
    validate_object_keys(s, {"hold_ms", "history_ms", "hold_words"}, "stability");
    if (s.contains("hold_ms"))
      c.hold_ms = s["hold_ms"].get<int>();
    if (s.contains("history_ms"))
      c.history_ms = s["history_ms"].get<int>();
    if (s.contains("hold_words"))
      c.hold_words = s["hold_words"].get<int>();
  }
  return c;
}

nlohmann::json config_to_json(Config const& config) {
  nlohmann::json j = nlohmann::json::object();
  j["schema_version"] = kConfigSchemaVersion;

  j["vad"] = {
      {"model", config.vad.model},
      {"path", config.vad.path},
      {"threshold", config.vad.threshold},
      {"hangover_ms", config.vad.hangover_ms},
  };

  j["asr"] = {
      {"model", config.asr.model},       {"family", config.asr.family},
      {"runtime", config.asr.runtime},   {"path", config.asr.path},
      {"chunk_ms", config.asr.chunk_ms}, {"beam_size", config.asr.beam_size},
  };

  j["llm"] = {
      {"enabled", config.llm.enabled},
      {"model", config.llm.model},
      {"path", config.llm.path},
      {"quantization", config.llm.quantization},
  };

  j["stability"] = {
      {"hold_ms", config.stability.hold_ms},
      {"history_ms", config.stability.history_ms},
      {"hold_words", config.stability.hold_words},
  };

  j["provider"] = config.provider;
  return j;
}

uint64_t fnv1a64(std::string const& text) {
  uint64_t hash = 14695981039346656037ULL;
  for (unsigned char byte : text) {
    hash ^= static_cast<uint64_t>(byte);
    hash *= 1099511628211ULL;
  }
  return hash;
}

}  // namespace

VadConfig::VadConfig() {
  auto const& defaults = default_values();
  model = defaults.vad.model;
  path = defaults.vad.path;
  threshold = defaults.vad.threshold;
  hangover_ms = defaults.vad.hangover_ms;
}

AsrConfig::AsrConfig() {
  auto const& defaults = default_values();
  model = defaults.asr.model;
  family = defaults.asr.family;
  runtime = defaults.asr.runtime;
  path = defaults.asr.path;
  chunk_ms = defaults.asr.chunk_ms;
  beam_size = defaults.asr.beam_size;
}

LlmConfig::LlmConfig() {
  auto const& defaults = default_values();
  enabled = defaults.llm.enabled;
  model = defaults.llm.model;
  path = defaults.llm.path;
  quantization = defaults.llm.quantization;
}

StabilityConfig::StabilityConfig() {
  auto const& defaults = default_values();
  hold_ms = defaults.stability.hold_ms;
  history_ms = defaults.stability.history_ms;
  hold_words = defaults.stability.hold_words;
}

Config::Config() : vad(), asr(), llm(), stability(), provider(default_values().provider) {}

Config load_config(std::string const& path) {
  std::ifstream f(path);
  if (!f.is_open()) {
    throw std::runtime_error("Cannot open config file: " + path);
  }
  std::stringstream ss;
  ss << f.rdbuf();
  return import_config_json(ss.str());
}

Config load_config_from_string(std::string const& json) {
  auto j = nlohmann::json::parse(json);
  validate_schema_version(j, false);
  return parse_config_json_object(j);
}

Config import_config_json(std::string const& json) {
  auto j = nlohmann::json::parse(json);
  validate_schema_version(j, true);
  return parse_config_json_object(j);
}

std::string export_effective_config_json(Config const& config) {
  return config_to_json(config).dump(2);
}

std::string effective_config_checksum_hex(Config const& config) {
  std::string const canonical_json = config_to_json(config).dump();
  uint64_t const hash = fnv1a64(canonical_json);
  std::ostringstream out;
  out << std::hex << std::setfill('0') << std::setw(16) << hash;
  return out.str();
}

}  // namespace ears
