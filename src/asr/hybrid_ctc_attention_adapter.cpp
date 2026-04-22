#if EARS_HAS_ONNX

#include "ears/asr/hybrid_ctc_attention_adapter.hpp"

#include <memory>
#include <string>
#include <utility>

#include "asr_adapter_json.hpp"
#include "ears/asr/ctc_asr_adapter.hpp"

namespace ears {

class HybridCtcAttentionAdapter::Impl {
public:
  explicit Impl(std::string const& model_path, int sample_rate, std::string provider)
      : provider_(std::move(provider)),
        ctc_(std::make_unique<CtcAsrAdapter>(model_path, sample_rate, provider_)) {}

  AsrResult recognize(float const* samples, size_t num_samples) {
    AsrResult out = ctc_->recognize(samples, num_samples);
    out.json = asr_json::build_delegate_metadata("hybrid_ctc_attention", "primary", "ctc",
                                                 provider_, "ctc_json", out.json);
    return out;
  }

private:
  std::string provider_;
  std::unique_ptr<CtcAsrAdapter> ctc_;
};

HybridCtcAttentionAdapter::HybridCtcAttentionAdapter(std::string const& model_path, int sample_rate,
                                                     std::string provider)
    : impl_(std::make_unique<Impl>(model_path, sample_rate, std::move(provider))) {}

HybridCtcAttentionAdapter::~HybridCtcAttentionAdapter() = default;

AsrResult HybridCtcAttentionAdapter::recognize(float const* samples, size_t num_samples) {
  return impl_->recognize(samples, num_samples);
}

}  // namespace ears

#endif  // EARS_HAS_ONNX
