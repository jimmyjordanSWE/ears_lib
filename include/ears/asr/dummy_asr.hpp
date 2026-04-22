#pragma once

#include "ears/asr.hpp"

namespace ears {

class DummyAsr : public IAutomaticSpeechRecognizer {
public:
  AsrResult recognize(float const* /*samples*/, size_t num_samples) override {
    AsrResult r;
    r.text = "[placeholder ASR: " + std::to_string(num_samples) + " samples]";
    r.json = "{}";
    r.confidence = 1.f;
    return r;
  }
};

}  // namespace ears
