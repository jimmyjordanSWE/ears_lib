#pragma once

#include <cstddef>
#include <string>

namespace ears {

struct AsrResult {
  std::string text;
  std::string json;        // n-best, confidences, etc.
  float confidence = 0.f;  // optional aggregate
};

struct IAutomaticSpeechRecognizer {
  virtual ~IAutomaticSpeechRecognizer() = default;
  virtual AsrResult recognize(float const* samples, size_t num_samples) = 0;
};

}  // namespace ears
