#include <atomic>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#include "ears/asr/native_runtime_asr_adapter.hpp"
#include "ears/config.hpp"
#include "ears/factory.hpp"

namespace ears {
namespace {

class ScopedEnvVar {
public:
  explicit ScopedEnvVar(char const* name) : name_(name), had_value_(env_exists(name)) {
    if (had_value_) {
      old_value_ = getenv_copy(name);
    }
  }

  ~ScopedEnvVar() {
    if (had_value_) {
      set(name_.c_str(), old_value_);
    } else {
      unset(name_.c_str());
    }
  }

  static void set(char const* name, std::string const& value) {
#ifdef _WIN32
    (void)_putenv_s(name, value.c_str());
#else
    (void)setenv(name, value.c_str(), 1);
#endif
  }

  static void unset(char const* name) {
#ifdef _WIN32
    (void)_putenv_s(name, "");
#else
    (void)unsetenv(name);
#endif
  }

private:
  static std::string getenv_copy(char const* name) {
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

  static bool env_exists(char const* name) {
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

  std::string name_;
  bool had_value_ = false;
  std::string old_value_;
};

std::string pick_compiled_native_runtime_or_empty() {
  std::vector<std::string> const runtimes = {"tensorrt", "openvino", "coreml", "qnn"};
  for (std::string const& runtime : runtimes) {
    if (has_asr_runtime_capability(runtime)) {
      return runtime;
    }
  }
  return "";
}

std::atomic<uint64_t>& script_counter() {
  static std::atomic<uint64_t> counter{1};
  return counter;
}

std::filesystem::path create_test_runner_script() {
  std::filesystem::path path =
      std::filesystem::temp_directory_path() /
      ("ears_native_runner_test_" + std::to_string(script_counter().fetch_add(1)) + ".ps1");

  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  out << "$pairs = @{}\n";
  out << "$probe = $false\n";
  out << "for ($i = 0; $i -lt $args.Count; $i++) {\n";
  out << "  $k = $args[$i]\n";
  out << "  if ($k -eq '--probe') { $probe = $true; continue }\n";
  out << "  if ($i + 1 -lt $args.Count) {\n";
  out << "    $pairs[$k] = $args[$i + 1]\n";
  out << "    $i++\n";
  out << "  }\n";
  out << "}\n";
  out << "if ($probe) {\n";
  out << "  if ($env:EARS_TEST_RUNNER_PROBE_FAIL -eq '1') { "
         "[Console]::Error.WriteLine('probe_fail'); "
         "exit 11 }\n";
  out << "  Write-Output '{\"probe\":\"ok\"}'\n";
  out << "  exit 0\n";
  out << "}\n";
  out << "if ($env:EARS_TEST_RUNNER_SLEEP_MS) { Start-Sleep -Milliseconds "
         "([int]$env:EARS_TEST_RUNNER_SLEEP_MS) }\n";
  out << "if ($env:EARS_TEST_RUNNER_STDOUT_BYTES) {\n";
  out << "  $n = [int]$env:EARS_TEST_RUNNER_STDOUT_BYTES\n";
  out << "  if ($n -gt 0) { Write-Output ('s' * $n) }\n";
  out << "}\n";
  out << "if ($env:EARS_TEST_RUNNER_STDERR_BYTES) {\n";
  out << "  $n = [int]$env:EARS_TEST_RUNNER_STDERR_BYTES\n";
  out << "  if ($n -gt 0) { [Console]::Error.WriteLine(('e' * $n)) }\n";
  out << "}\n";
  out << "if ($env:EARS_TEST_RUNNER_EXIT_CODE) {\n";
  out << "  $code = [int]$env:EARS_TEST_RUNNER_EXIT_CODE\n";
  out << "  if ($code -ne 0) { exit $code }\n";
  out << "}\n";
  out << "$output = $pairs['--output-text']\n";
  out << "if (-not $output) { exit 10 }\n";
  out << "Set-Content -Path $output -Value 'native runner transcript' -NoNewline\n";
  out << "exit 0\n";
  return path;
}

std::string runner_command_for_script(std::filesystem::path const& script_path) {
  return "powershell.exe -NoProfile -ExecutionPolicy Bypass -File \"" + script_path.string() + "\"";
}

std::string runtime_timeout_env_name(std::string const& runtime) {
  if (runtime == "tensorrt") {
    return "EARS_TENSORRT_ASR_RUNNER_TIMEOUT_MS";
  }
  if (runtime == "openvino") {
    return "EARS_OPENVINO_ASR_RUNNER_TIMEOUT_MS";
  }
  if (runtime == "coreml") {
    return "EARS_COREML_ASR_RUNNER_TIMEOUT_MS";
  }
  return "EARS_QNN_ASR_RUNNER_TIMEOUT_MS";
}

class TestModelAsr final : public IAutomaticSpeechRecognizer {
public:
  AsrResult recognize(float const* /*samples*/, size_t /*num_samples*/) override {
    AsrResult result;
    result.text = "model_fallback";
    result.confidence = 0.5f;
    result.json = "{\"source\":\"model\"}";
    return result;
  }
};

TEST(NativeRuntimeRunnerTest, ProbeRunnerNotConfigured_ReturnsStableStatus) {
  std::string const runtime = pick_compiled_native_runtime_or_empty();
  if (runtime.empty()) {
    GTEST_SKIP() << "No compiled native runtime in this build profile.";
  }

  ScopedEnvVar runner_env(native_runtime_runner_env_name(runtime).c_str());
  ScopedEnvVar::unset(native_runtime_runner_env_name(runtime).c_str());

  NativeRuntimeRunnerProbe const probe = probe_native_runtime_runner(runtime);
  EXPECT_FALSE(probe.configured);
  EXPECT_FALSE(probe.healthy);
  EXPECT_EQ(probe.status_code, "runner_not_configured");
}

TEST(NativeRuntimeRunnerTest, RecognizeMapsRunnerExitCodeToStableLibraryCode) {
#ifndef _WIN32
  GTEST_SKIP() << "Runner process test currently uses PowerShell script harness on Windows.";
#else
  std::string const runtime = pick_compiled_native_runtime_or_empty();
  if (runtime.empty()) {
    GTEST_SKIP() << "No compiled native runtime in this build profile.";
  }

  std::filesystem::path const script_path = create_test_runner_script();
  ScopedEnvVar runner_env(native_runtime_runner_env_name(runtime).c_str());
  ScopedEnvVar exit_env("EARS_TEST_RUNNER_EXIT_CODE");
  ScopedEnvVar sleep_env("EARS_TEST_RUNNER_SLEEP_MS");
  ScopedEnvVar stdout_env("EARS_TEST_RUNNER_STDOUT_BYTES");
  ScopedEnvVar stderr_env("EARS_TEST_RUNNER_STDERR_BYTES");

  ScopedEnvVar::set(native_runtime_runner_env_name(runtime).c_str(),
                    runner_command_for_script(script_path));
  ScopedEnvVar::set("EARS_TEST_RUNNER_EXIT_CODE", "14");
  ScopedEnvVar::unset("EARS_TEST_RUNNER_SLEEP_MS");
  ScopedEnvVar::unset("EARS_TEST_RUNNER_STDOUT_BYTES");
  ScopedEnvVar::unset("EARS_TEST_RUNNER_STDERR_BYTES");

  NativeRuntimeAsrAdapter adapter(runtime, "ctc", "models/fake.onnx", 16000);
  std::vector<float> samples(256, 0.0f);
  AsrResult const result = adapter.recognize(samples.data(), samples.size());
  nlohmann::json const meta = nlohmann::json::parse(result.json);
  EXPECT_EQ(meta["decode"], "runner_failed");
  EXPECT_EQ(meta["code"], "decode_failed");
  EXPECT_FLOAT_EQ(result.confidence, 0.0f);
  std::error_code ec;
  std::filesystem::remove(script_path, ec);
#endif
}

TEST(NativeRuntimeRunnerTest, RecognizeTimeoutAndBoundedLogs_AreReportedInMetadata) {
#ifndef _WIN32
  GTEST_SKIP() << "Runner process test currently uses PowerShell script harness on Windows.";
#else
  std::string const runtime = pick_compiled_native_runtime_or_empty();
  if (runtime.empty()) {
    GTEST_SKIP() << "No compiled native runtime in this build profile.";
  }

  std::filesystem::path const script_path = create_test_runner_script();
  std::string const timeout_env_name = runtime_timeout_env_name(runtime);

  ScopedEnvVar runner_env(native_runtime_runner_env_name(runtime).c_str());
  ScopedEnvVar timeout_env(timeout_env_name.c_str());
  ScopedEnvVar output_cap_env("EARS_NATIVE_RUNNER_OUTPUT_MAX_BYTES");
  ScopedEnvVar exit_env("EARS_TEST_RUNNER_EXIT_CODE");
  ScopedEnvVar sleep_env("EARS_TEST_RUNNER_SLEEP_MS");
  ScopedEnvVar stdout_env("EARS_TEST_RUNNER_STDOUT_BYTES");
  ScopedEnvVar stderr_env("EARS_TEST_RUNNER_STDERR_BYTES");

  ScopedEnvVar::set(native_runtime_runner_env_name(runtime).c_str(),
                    runner_command_for_script(script_path));
  ScopedEnvVar::set(timeout_env_name.c_str(), "100");
  ScopedEnvVar::set("EARS_NATIVE_RUNNER_OUTPUT_MAX_BYTES", "256");
  ScopedEnvVar::set("EARS_TEST_RUNNER_SLEEP_MS", "500");
  ScopedEnvVar::set("EARS_TEST_RUNNER_STDOUT_BYTES", "4096");
  ScopedEnvVar::set("EARS_TEST_RUNNER_STDERR_BYTES", "4096");
  ScopedEnvVar::unset("EARS_TEST_RUNNER_EXIT_CODE");

  NativeRuntimeAsrAdapter adapter(runtime, "ctc", "models/fake.onnx", 16000);
  std::vector<float> samples(256, 0.0f);
  AsrResult const timeout_result = adapter.recognize(samples.data(), samples.size());
  nlohmann::json const timeout_meta = nlohmann::json::parse(timeout_result.json);
  EXPECT_EQ(timeout_meta["decode"], "runner_timeout");
  EXPECT_EQ(timeout_meta["code"], "runner_timeout");

  ScopedEnvVar::set(timeout_env_name.c_str(), "3000");
  ScopedEnvVar::set("EARS_TEST_RUNNER_SLEEP_MS", "0");
  AsrResult const success_result = adapter.recognize(samples.data(), samples.size());
  nlohmann::json const success_meta = nlohmann::json::parse(success_result.json);

  EXPECT_EQ(success_meta["decode"], "ok");
  EXPECT_TRUE(success_meta["runner_stdout_truncated"].get<bool>());
  EXPECT_TRUE(success_meta["runner_stderr_truncated"].get<bool>());
  EXPECT_LE(success_meta["runner_stdout_tail"].get<std::string>().size(), 256u);
  EXPECT_LE(success_meta["runner_stderr_tail"].get<std::string>().size(), 256u);

  std::error_code ec;
  std::filesystem::remove(script_path, ec);
#endif
}

TEST(NativeRuntimeRunnerTest, FactoryFallbackPolicy_ModelFallbackWhenRunnerUnavailable) {
  std::string const runtime = pick_compiled_native_runtime_or_empty();
  if (runtime.empty()) {
    GTEST_SKIP() << "No compiled native runtime in this build profile.";
  }

  ScopedEnvVar policy_env("EARS_NATIVE_RUNNER_UNAVAILABLE_POLICY");
  ScopedEnvVar runner_env(native_runtime_runner_env_name(runtime).c_str());
  ScopedEnvVar::set("EARS_NATIVE_RUNNER_UNAVAILABLE_POLICY", "model");
  ScopedEnvVar::unset(native_runtime_runner_env_name(runtime).c_str());

  reset_factory_registry();
  EXPECT_TRUE(register_asr_factory("runner_fallback_model",
                                   [](Config const&) { return std::make_unique<TestModelAsr>(); }));

  Config config;
  config.asr.model = "runner_fallback_model";
  config.asr.family = "ctc";
  config.asr.runtime = runtime;
  config.provider = "cpu";

  auto asr = create_asr(config);
  ASSERT_NE(asr, nullptr);
  AsrResult const result = asr->recognize(nullptr, 0);
  EXPECT_EQ(result.text, "model_fallback");
}

TEST(NativeRuntimeRunnerTest, FactoryFallbackPolicy_ErrorWhenConfiguredToFail) {
  std::string const runtime = pick_compiled_native_runtime_or_empty();
  if (runtime.empty()) {
    GTEST_SKIP() << "No compiled native runtime in this build profile.";
  }

  ScopedEnvVar policy_env("EARS_NATIVE_RUNNER_UNAVAILABLE_POLICY");
  ScopedEnvVar runner_env(native_runtime_runner_env_name(runtime).c_str());
  ScopedEnvVar::set("EARS_NATIVE_RUNNER_UNAVAILABLE_POLICY", "error");
  ScopedEnvVar::unset(native_runtime_runner_env_name(runtime).c_str());

  Config config;
  config.asr.model = "runner_fallback_model";
  config.asr.family = "ctc";
  config.asr.runtime = runtime;

  auto asr = create_asr(config);
  ASSERT_NE(asr, nullptr);
  AsrResult const result = asr->recognize(nullptr, 0);
  EXPECT_TRUE(result.json.find("\"code\":\"runner_unavailable\"") != std::string::npos);
}

}  // namespace
}  // namespace ears
