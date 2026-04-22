#include "ears/asr/openvino_asr_adapter.hpp"

#include <memory>
#include <string>
#include <utility>

#include "ears/asr/native_runtime_asr_adapter.hpp"

namespace ears {

class OpenVinoAsrAdapter::Impl {
public:
  Impl(std::string family_id, std::string const& model_path, int sample_rate)
      : inner_(std::make_unique<NativeRuntimeAsrAdapter>("openvino", std::move(family_id),
                                                         model_path, sample_rate)) {}

  AsrResult recognize(float const* samples, size_t num_samples) {
    return inner_->recognize(samples, num_samples);
  }

private:
  std::unique_ptr<NativeRuntimeAsrAdapter> inner_;
};

OpenVinoAsrAdapter::OpenVinoAsrAdapter(std::string family_id, std::string const& model_path,
                                       int sample_rate)
    : impl_(std::make_unique<Impl>(std::move(family_id), model_path, sample_rate)) {}

OpenVinoAsrAdapter::~OpenVinoAsrAdapter() = default;

AsrResult OpenVinoAsrAdapter::recognize(float const* samples, size_t num_samples) {
  return impl_->recognize(samples, num_samples);
}

}  // namespace ears
