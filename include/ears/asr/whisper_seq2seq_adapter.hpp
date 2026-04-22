#pragma once

#include <memory>
#include <string>

#include "ears/asr.hpp"

namespace ears {

class WhisperSeq2SeqAdapter : public IAutomaticSpeechRecognizer {
public:
  explicit WhisperSeq2SeqAdapter(std::string const& model_path, int sample_rate = 16000,
                                 std::string provider = "auto");
  ~WhisperSeq2SeqAdapter() override;

  AsrResult recognize(float const* samples, size_t num_samples) override;

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace ears
