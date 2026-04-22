#pragma once

#include <memory>
#include <string>

#include "ears/asr.hpp"

namespace ears {

class MoonshineAsr : public IAutomaticSpeechRecognizer {
public:
  explicit MoonshineAsr(std::string const& model_path, int sample_rate = 16000);
  ~MoonshineAsr() override;

  AsrResult recognize(float const* samples, size_t num_samples) override;

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace ears
