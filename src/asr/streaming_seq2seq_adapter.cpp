#if EARS_HAS_ONNX

#include "ears/asr/streaming_seq2seq_adapter.hpp"

#include <memory>
#include <string>
#include <utility>

#include "asr_adapter_json.hpp"
#include "ears/asr/whisper_seq2seq_adapter.hpp"

namespace ears {

class StreamingSeq2SeqAdapter::Impl {
public:
  explicit Impl(std::string const& model_path, int sample_rate, std::string provider)
      : provider_(std::move(provider)),
        inner_(std::make_unique<WhisperSeq2SeqAdapter>(model_path, sample_rate, provider_)) {}

  AsrResult recognize(float const* samples, size_t num_samples) {
    AsrResult out = inner_->recognize(samples, num_samples);
    out.json = asr_json::build_delegate_metadata("streaming_seq2seq", "delegate", "whisper_seq2seq",
                                                 provider_, "delegate_json", out.json);
    return out;
  }

private:
  std::string provider_;
  std::unique_ptr<WhisperSeq2SeqAdapter> inner_;
};

StreamingSeq2SeqAdapter::StreamingSeq2SeqAdapter(std::string const& model_path, int sample_rate,
                                                 std::string provider)
    : impl_(std::make_unique<Impl>(model_path, sample_rate, std::move(provider))) {}

StreamingSeq2SeqAdapter::~StreamingSeq2SeqAdapter() = default;

AsrResult StreamingSeq2SeqAdapter::recognize(float const* samples, size_t num_samples) {
  return impl_->recognize(samples, num_samples);
}

}  // namespace ears

#endif  // EARS_HAS_ONNX
