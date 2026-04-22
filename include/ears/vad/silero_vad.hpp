#pragma once

#include <memory>
#include <string>

#include "ears/vad.hpp"

namespace ears {

class SileroVad : public IVoiceActivityDetector {
public:
  explicit SileroVad(std::string const& model_path, int sample_rate = 16000);
  ~SileroVad() override;

  float get_speech_probability(float const* samples, size_t num_samples) override;

  void reset();

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace ears
