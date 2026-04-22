#pragma once

#include <cstddef>
#include <cstdint>

namespace ears {

struct IVoiceActivityDetector {
  virtual ~IVoiceActivityDetector() = default;
  virtual float get_speech_probability(float const* samples, size_t num_samples) = 0;
};

}  // namespace ears
