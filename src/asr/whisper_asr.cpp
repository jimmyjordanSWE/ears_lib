#if EARS_HAS_ONNX

#include "ears/asr/whisper_asr.hpp"

#include <memory>
#include <string>
#include <utility>

#include "asr_adapter_json.hpp"
#include "ears/asr/whisper_seq2seq_adapter.hpp"

namespace ears {

class WhisperAsr::Impl {
public:
  explicit Impl(std::string model_path, int sample_rate)
      : inner_(
            std::make_unique<WhisperSeq2SeqAdapter>(std::move(model_path), sample_rate, "auto")) {}

  AsrResult recognize(float const* samples, size_t num_samples) {
    AsrResult out = inner_->recognize(samples, num_samples);
    out.json = asr_json::build_delegate_metadata("whisper", "delegate", "whisper_seq2seq", "auto",
                                                 "delegate_json", out.json);
    return out;
  }

private:
  std::unique_ptr<WhisperSeq2SeqAdapter> inner_;
};

WhisperAsr::WhisperAsr(std::string const& model_path, int sample_rate)
    : impl_(std::make_unique<Impl>(model_path, sample_rate)) {}

WhisperAsr::~WhisperAsr() = default;

AsrResult WhisperAsr::recognize(float const* samples, size_t num_samples) {
  return impl_->recognize(samples, num_samples);
}

}  // namespace ears

#endif  // EARS_HAS_ONNX
