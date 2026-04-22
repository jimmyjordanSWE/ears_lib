#include "ears/config.hpp"

#include <cstdlib>
#include <fstream>
#include <gtest/gtest.h>
#include <stdexcept>

namespace ears {
namespace {

void expect_config_eq(Config const& lhs, Config const& rhs) {
  EXPECT_EQ(lhs.vad.model, rhs.vad.model);
  EXPECT_EQ(lhs.vad.path, rhs.vad.path);
  EXPECT_FLOAT_EQ(lhs.vad.threshold, rhs.vad.threshold);
  EXPECT_EQ(lhs.vad.hangover_ms, rhs.vad.hangover_ms);

  EXPECT_EQ(lhs.asr.model, rhs.asr.model);
  EXPECT_EQ(lhs.asr.family, rhs.asr.family);
  EXPECT_EQ(lhs.asr.runtime, rhs.asr.runtime);
  EXPECT_EQ(lhs.asr.path, rhs.asr.path);
  EXPECT_EQ(lhs.asr.chunk_ms, rhs.asr.chunk_ms);
  EXPECT_EQ(lhs.asr.beam_size, rhs.asr.beam_size);

  EXPECT_EQ(lhs.llm.enabled, rhs.llm.enabled);
  EXPECT_EQ(lhs.llm.model, rhs.llm.model);
  EXPECT_EQ(lhs.llm.path, rhs.llm.path);
  EXPECT_EQ(lhs.llm.quantization, rhs.llm.quantization);

  EXPECT_EQ(lhs.stability.hold_ms, rhs.stability.hold_ms);
  EXPECT_EQ(lhs.stability.history_ms, rhs.stability.history_ms);
  EXPECT_EQ(lhs.stability.hold_words, rhs.stability.hold_words);

  EXPECT_EQ(lhs.provider, rhs.provider);
}

std::string getenv_copy_for_test(char const* name) {
#ifdef _WIN32
  char* value = nullptr;
  size_t value_len = 0;
  if (_dupenv_s(&value, &value_len, name) != 0 || value == nullptr) {
    return "";
  }
  std::string out(value);
  free(value);
  return out;
#else
  char const* value = std::getenv(name);
  return value == nullptr ? std::string() : std::string(value);
#endif
}

bool env_exists_for_test(char const* name) {
#ifdef _WIN32
  char* value = nullptr;
  size_t value_len = 0;
  if (_dupenv_s(&value, &value_len, name) != 0 || value == nullptr) {
    return false;
  }
  free(value);
  return true;
#else
  return std::getenv(name) != nullptr;
#endif
}

void set_env_for_test(char const* name, std::string const& value) {
#ifdef _WIN32
  (void)_putenv_s(name, value.c_str());
#else
  (void)setenv(name, value.c_str(), 1);
#endif
}

void unset_env_for_test(char const* name) {
#ifdef _WIN32
  (void)_putenv_s(name, "");
#else
  (void)unsetenv(name);
#endif
}

TEST(ConfigTest, LoadConfigFromString_EmptyJson_UsesDefaults) {
  std::string json = "{}";
  Config c = load_config_from_string(json);

  EXPECT_EQ(c.vad.model, "silero");
  EXPECT_EQ(c.vad.path, "models/silero_v5.onnx");
  EXPECT_FLOAT_EQ(c.vad.threshold, 0.8f);
  EXPECT_EQ(c.vad.hangover_ms, 400);

  EXPECT_EQ(c.asr.model, "moonshine");
  EXPECT_EQ(c.asr.family, "");
  EXPECT_EQ(c.asr.runtime, "auto");
  EXPECT_EQ(c.asr.path, "models/moonshine_tiny.onnx");
  EXPECT_EQ(c.asr.chunk_ms, 500);
  EXPECT_EQ(c.asr.beam_size, 1);

  EXPECT_EQ(c.llm.model, "smollm2");
  EXPECT_EQ(c.llm.path, "models/smollm2_360m.onnx");
  EXPECT_EQ(c.llm.quantization, "int4");

  EXPECT_EQ(c.stability.hold_ms, 250);
  EXPECT_EQ(c.stability.history_ms, 1000);
  EXPECT_EQ(c.stability.hold_words, 3);

  EXPECT_EQ(c.provider, "cpu");
}

TEST(ConfigTest, LoadConfigFromString_WithVad_OverridesDefaults) {
  std::string json = R"({
    "vad": {
      "model": "custom",
      "path": "models/custom_vad.onnx",
      "threshold": 0.9,
      "hangover_ms": 500
    }
  })";
  Config c = load_config_from_string(json);

  EXPECT_EQ(c.vad.model, "custom");
  EXPECT_EQ(c.vad.path, "models/custom_vad.onnx");
  EXPECT_FLOAT_EQ(c.vad.threshold, 0.9f);
  EXPECT_EQ(c.vad.hangover_ms, 500);
}

TEST(ConfigTest, LoadConfigFromString_WithAsr_OverridesDefaults) {
  std::string json = R"({
    "asr": {
      "model": "whisper",
      "family": "whisper_seq2seq",
      "runtime": "onnx",
      "path": "models/whisper.onnx",
      "chunk_ms": 250,
      "beam_size": 4
    }
  })";
  Config c = load_config_from_string(json);

  EXPECT_EQ(c.asr.model, "whisper");
  EXPECT_EQ(c.asr.family, "whisper_seq2seq");
  EXPECT_EQ(c.asr.runtime, "onnx");
  EXPECT_EQ(c.asr.path, "models/whisper.onnx");
  EXPECT_EQ(c.asr.chunk_ms, 250);
  EXPECT_EQ(c.asr.beam_size, 4);
}

TEST(ConfigTest, LoadConfigFromString_WithLlm_OverridesDefaults) {
  std::string json = R"({
    "llm": {
      "model": "gemma",
      "path": "models/gemma.onnx",
      "quantization": "int8"
    }
  })";
  Config c = load_config_from_string(json);

  EXPECT_EQ(c.llm.model, "gemma");
  EXPECT_EQ(c.llm.path, "models/gemma.onnx");
  EXPECT_EQ(c.llm.quantization, "int8");
}

TEST(ConfigTest, LoadConfigFromString_WithStability_OverridesDefaults) {
  std::string json = R"({
    "stability": {
      "hold_ms": 300,
      "history_ms": 1500,
      "hold_words": 5
    }
  })";
  Config c = load_config_from_string(json);

  EXPECT_EQ(c.stability.hold_ms, 300);
  EXPECT_EQ(c.stability.history_ms, 1500);
  EXPECT_EQ(c.stability.hold_words, 5);
}

TEST(ConfigTest, LoadConfigFromString_WithProvider_SetsProvider) {
  std::string json = R"({"provider": "cuda"})";
  Config c = load_config_from_string(json);
  EXPECT_EQ(c.provider, "cuda");
}

TEST(ConfigTest, LoadConfigFromString_InvalidJson_Throws) {
  EXPECT_THROW(load_config_from_string("{ invalid }"), std::exception);
}

TEST(ConfigTest, LoadConfigFromString_UnknownTopLevelKey_Throws) {
  EXPECT_THROW(load_config_from_string(R"({"unknown_key": 1})"), std::runtime_error);
}

TEST(ConfigTest, LoadConfigFromString_UnknownSectionKey_Throws) {
  EXPECT_THROW(load_config_from_string(R"({"asr": {"unknown_knob": 2}})"), std::runtime_error);
}

TEST(ConfigTest, LoadConfigFromString_UnknownVadKey_Throws) {
  EXPECT_THROW(load_config_from_string(R"({"vad": {"unknown_knob": 2}})"), std::runtime_error);
}

TEST(ConfigTest, LoadConfigFromString_UnknownLlmKey_Throws) {
  EXPECT_THROW(load_config_from_string(R"({"llm": {"unknown_knob": 2}})"), std::runtime_error);
}

TEST(ConfigTest, LoadConfigFromString_UnknownStabilityKey_Throws) {
  EXPECT_THROW(load_config_from_string(R"({"stability": {"unknown_knob": 2}})"),
               std::runtime_error);
}

TEST(ConfigTest, LoadConfig_FileNotFound_Throws) {
  EXPECT_THROW(load_config("nonexistent_file_that_does_not_exist.json"), std::runtime_error);
}

TEST(ConfigTest, LoadConfig_ValidFile_LoadsSuccessfully) {
  std::string tmp_path = "ears_config_test_temp.json";
  std::ofstream f(tmp_path);
  ASSERT_TRUE(f.is_open());
  f << R"({"schema_version": 1, "provider": "directml"})";
  f.close();

  Config c = load_config(tmp_path);
  EXPECT_EQ(c.provider, "directml");

  std::remove(tmp_path.c_str());
}

TEST(ConfigTest, ExportEffectiveConfigJson_EmitsFullResolvedConfig) {
  Config config;
  std::string const exported_json = export_effective_config_json(config);
  Config const imported = import_config_json(exported_json);
  expect_config_eq(config, imported);
}

TEST(ConfigTest, ImportConfigJson_RoundTripPreservesCustomValues) {
  Config config;
  config.vad.model = "custom-vad";
  config.vad.path = "models/custom_vad.onnx";
  config.vad.threshold = 0.42f;
  config.vad.hangover_ms = 123;
  config.asr.model = "custom-asr";
  config.asr.family = "ctc";
  config.asr.runtime = "onnx_runtime";
  config.asr.path = "models/custom_asr.onnx";
  config.asr.chunk_ms = 333;
  config.asr.beam_size = 7;
  config.llm.enabled = true;
  config.llm.model = "custom-llm";
  config.llm.path = "models/custom_llm.onnx";
  config.llm.quantization = "int8";
  config.stability.hold_ms = 12;
  config.stability.history_ms = 34;
  config.stability.hold_words = 2;
  config.provider = "directml";

  std::string const exported_json = export_effective_config_json(config);
  Config const imported = import_config_json(exported_json);
  expect_config_eq(config, imported);
}

TEST(ConfigTest, ImportConfigJson_RejectsUnknownKeys) {
  EXPECT_THROW(import_config_json(R"({"schema_version": 1, "asr": {"unknown_knob": 2}})"),
               std::runtime_error);
}

TEST(ConfigTest, ImportConfigJson_MissingSchemaVersion_Throws) {
  EXPECT_THROW(import_config_json(R"({"provider":"cpu"})"), std::runtime_error);
}

TEST(ConfigTest, ImportConfigJson_UnsupportedSchemaVersion_Throws) {
  EXPECT_THROW(import_config_json(R"({"schema_version": 999, "provider":"cpu"})"),
               std::runtime_error);
}

TEST(ConfigTest, EffectiveConfigChecksumHex_IsStableForSameConfigAndChangesOnEdit) {
  Config config;
  std::string const checksum_a = effective_config_checksum_hex(config);
  std::string const checksum_b = effective_config_checksum_hex(config);
  EXPECT_EQ(checksum_a, checksum_b);

  config.asr.chunk_ms += 1;
  std::string const checksum_c = effective_config_checksum_hex(config);
  EXPECT_NE(checksum_a, checksum_c);
}

TEST(ConfigTest, LoadConfigFromString_ValueBounds_Throws) {
  EXPECT_THROW(load_config_from_string(R"({"vad":{"threshold":-0.01}})"), std::runtime_error);
  EXPECT_THROW(load_config_from_string(R"({"vad":{"threshold":1.01}})"), std::runtime_error);
  EXPECT_THROW(load_config_from_string(R"({"vad":{"hangover_ms":-1}})"), std::runtime_error);
  EXPECT_THROW(load_config_from_string(R"({"asr":{"chunk_ms":0}})"), std::runtime_error);
  EXPECT_THROW(load_config_from_string(R"({"asr":{"beam_size":0}})"), std::runtime_error);
  EXPECT_THROW(load_config_from_string(R"({"stability":{"hold_ms":-1}})"), std::runtime_error);
  EXPECT_THROW(load_config_from_string(R"({"stability":{"history_ms":-1}})"), std::runtime_error);
  EXPECT_THROW(load_config_from_string(R"({"stability":{"hold_words":-1}})"), std::runtime_error);
}

TEST(ConfigTest, LoadConfigFromString_InvalidTypes_Throws) {
  EXPECT_THROW(load_config_from_string(R"({"provider":123})"), std::exception);
  EXPECT_THROW(load_config_from_string(R"({"vad":{"threshold":"bad"}})"), std::exception);
  EXPECT_THROW(load_config_from_string(R"({"asr":{"chunk_ms":"bad"}})"), std::exception);
  EXPECT_THROW(load_config_from_string(R"({"llm":{"enabled":"bad"}})"), std::exception);
}

TEST(ConfigTest, LoadConfigFromString_MalformedRuntimeProviderCombo_Throws) {
  EXPECT_THROW(load_config_from_string(R"({"asr":{"runtime":"tensorrt"},"provider":"directml"})"),
               std::runtime_error);
  EXPECT_THROW(load_config_from_string(R"({"asr":{"runtime":"qnn"},"provider":"cuda"})"),
               std::runtime_error);
}

TEST(ConfigTest, LoadConfigFromString_ValidRuntimeProviderCombo_Passes) {
  EXPECT_NO_THROW(load_config_from_string(R"({"asr":{"runtime":"onnx"},"provider":"dml"})"));
  EXPECT_NO_THROW(load_config_from_string(R"({"asr":{"runtime":"tensorrt"},"provider":"cuda"})"));
}

TEST(ConfigTest, StrictStartupMode_MissingMasterConfig_Throws) {
  bool const had_strict = env_exists_for_test("EARS_STRICT_STARTUP");
  bool const had_master_path = env_exists_for_test("EARS_MASTER_CONFIG_PATH");
  std::string const old_strict = getenv_copy_for_test("EARS_STRICT_STARTUP");
  std::string const old_master_path = getenv_copy_for_test("EARS_MASTER_CONFIG_PATH");

  set_env_for_test("EARS_STRICT_STARTUP", "1");
  set_env_for_test("EARS_MASTER_CONFIG_PATH", "missing_master_config_for_test.json");

  EXPECT_THROW(
      {
        Config strict_config;
        (void)strict_config;
      },
      std::runtime_error);

  if (had_strict) {
    set_env_for_test("EARS_STRICT_STARTUP", old_strict);
  } else {
    unset_env_for_test("EARS_STRICT_STARTUP");
  }
  if (had_master_path) {
    set_env_for_test("EARS_MASTER_CONFIG_PATH", old_master_path);
  } else {
    unset_env_for_test("EARS_MASTER_CONFIG_PATH");
  }
}

}  // namespace
}  // namespace ears
