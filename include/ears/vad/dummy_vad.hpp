#pragma once

#include "ears/vad.hpp"

namespace ears {

class DummyVad : public IVoiceActivityDetector {
public:
  explicit DummyVad(float fixed_probability = 1.f) : prob_(fixed_probability) {}
  float get_speech_probability(float const* /*samples*/, size_t /*num_samples*/) override {
    return prob_;
  }

private:
  float prob_;
};

}  // namespace ears
