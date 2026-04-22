#if EARS_HAS_ONNX

#include "ears/asr/moonshine_asr.hpp"

#include <onnxruntime_cxx_api.h>
#include <string>

#include "ears/internal/unicode_utils.hpp"

namespace ears {

class MoonshineAsr::Impl {
public:
  explicit Impl(std::string const& model_path) : env_(ORT_LOGGING_LEVEL_WARNING, "ears_moonshine") {
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(1);
    opts.SetInterOpNumThreads(1);
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    std::wstring wpath = internal::to_wide_utf8(model_path);
    session_ = std::make_unique<Ort::Session>(env_, wpath.c_str(), opts);
  }

  AsrResult recognize(float const* samples, size_t num_samples) {
    (void)samples;
    (void)num_samples;
    AsrResult r;
    r.text = "[moonshine: " + std::to_string(num_samples) + " samples]";
    r.json = "{}";
    r.confidence = 0.9f;
    return r;
  }

private:
  Ort::Env env_;
  std::unique_ptr<Ort::Session> session_;
};

MoonshineAsr::MoonshineAsr(std::string const& model_path, int sample_rate)
    : impl_(std::make_unique<Impl>(model_path)) {
  (void)sample_rate;
}

MoonshineAsr::~MoonshineAsr() = default;

AsrResult MoonshineAsr::recognize(float const* samples, size_t num_samples) {
  return impl_->recognize(samples, num_samples);
}

}  // namespace ears

#endif  // EARS_HAS_ONNX
