#pragma once

#include <memory>
#include <string>

#include "ears/asr.hpp"

namespace ears {

struct NativeRuntimeRunnerProbe {
  std::string runtime;
  std::string runner_env;
  std::string runner_command;
  std::string status_code;
  std::string message;
  std::string stdout_tail;
  std::string stderr_tail;
  int exit_code = 0;
  bool configured = false;
  bool healthy = false;
  bool timed_out = false;
  bool stdout_truncated = false;
  bool stderr_truncated = false;
};

std::string native_runtime_runner_env_name(std::string const& runtime_id);
bool is_native_runtime_id(std::string const& runtime_id);
NativeRuntimeRunnerProbe probe_native_runtime_runner(std::string const& runtime_id);

class NativeRuntimeAsrAdapter : public IAutomaticSpeechRecognizer {
public:
  NativeRuntimeAsrAdapter(std::string runtime_id, std::string family_id,
                          std::string const& model_path, int sample_rate = 16000);
  ~NativeRuntimeAsrAdapter() override;

  AsrResult recognize(float const* samples, size_t num_samples) override;

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace ears
