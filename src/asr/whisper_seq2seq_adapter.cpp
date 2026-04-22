#if EARS_HAS_ONNX

#include "ears/asr/whisper_seq2seq_adapter.hpp"

#include <string>
#include <utility>

#include "asr_adapter_json.hpp"
#include "ears/transcribe.hpp"

namespace ears {

class WhisperSeq2SeqAdapter::Impl {
public:
  explicit Impl(std::string model_path, int sample_rate, std::string provider)
      : model_path_(std::move(model_path)),
        sample_rate_(sample_rate),
        provider_(std::move(provider)) {}

  AsrResult recognize(float const* samples, size_t num_samples) {
    AsrResult result;
    try {
      std::string text =
          transcribe_audio(samples, num_samples, sample_rate_, model_path_, provider_);
      if (!text.empty()) {
        result.text = text;
        result.json = asr_json::build_status_metadata("whisper_seq2seq", "onnx_runtime", provider_,
                                                      "success");
        result.confidence = 0.9f;
        return result;
      }
    } catch (...) {
      result.confidence = 0.0f;
    }

    result.text = "[whisper_seq2seq: " + std::to_string(num_samples) + " samples]";
    result.json =
        asr_json::build_status_metadata("whisper_seq2seq", "onnx_runtime", provider_, "fallback");
    result.confidence = 0.5f;
    return result;
  }

private:
  std::string model_path_;
  int sample_rate_ = 16000;
  std::string provider_ = "auto";
};

WhisperSeq2SeqAdapter::WhisperSeq2SeqAdapter(std::string const& model_path, int sample_rate,
                                             std::string provider)
    : impl_(std::make_unique<Impl>(model_path, sample_rate, std::move(provider))) {}

WhisperSeq2SeqAdapter::~WhisperSeq2SeqAdapter() = default;

AsrResult WhisperSeq2SeqAdapter::recognize(float const* samples, size_t num_samples) {
  return impl_->recognize(samples, num_samples);
}

}  // namespace ears

#endif  // EARS_HAS_ONNX
