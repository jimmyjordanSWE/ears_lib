#if EARS_HAS_ONNX

#include "ears/llm/smollm_rectifier.hpp"

#include <onnxruntime_cxx_api.h>
#include <string>

#include "ears/internal/unicode_utils.hpp"

namespace ears {

class SmolLMRectifier::Impl {
public:
  explicit Impl(std::string const& model_path) : env_(ORT_LOGGING_LEVEL_WARNING, "ears_smollm") {
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(1);
    opts.SetInterOpNumThreads(1);
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    std::wstring wpath = internal::to_wide_utf8(model_path);
    session_ = std::make_unique<Ort::Session>(env_, wpath.c_str(), opts);
  }

  std::string rectify(std::string const& phonetic_text, std::string const& app_context,
                      std::string const& history) {
    (void)app_context;
    (void)history;
    return phonetic_text;  // pass-through until tokenizer added
  }

private:
  Ort::Env env_;
  std::unique_ptr<Ort::Session> session_;
};

SmolLMRectifier::SmolLMRectifier(std::string const& model_path)
    : impl_(std::make_unique<Impl>(model_path)) {}

SmolLMRectifier::~SmolLMRectifier() = default;

std::string SmolLMRectifier::rectify(std::string const& phonetic_text,
                                     std::string const& app_context, std::string const& history) {
  return impl_->rectify(phonetic_text, app_context, history);
}

}  // namespace ears

#endif  // EARS_HAS_ONNX
