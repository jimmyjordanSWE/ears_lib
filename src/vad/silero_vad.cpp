#if EARS_HAS_ONNX

#include "ears/vad/silero_vad.hpp"

#include <algorithm>
#include <cstring>
#include <onnxruntime_cxx_api.h>
#include <string>
#include <vector>

#include "ears/internal/unicode_utils.hpp"

namespace ears {

class SileroVad::Impl {
public:
  explicit Impl(std::string const& model_path, int sample_rate)
      : sample_rate_(sample_rate),
        window_size_samples_(32 * (sample_rate / 1000)),  // 32 ms
        context_samples_(64),
        effective_window_size_(window_size_samples_ + context_samples_),
        env_(ORT_LOGGING_LEVEL_WARNING, "ears_silero") {
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(1);
    opts.SetInterOpNumThreads(1);
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    std::wstring wpath = internal::to_wide_utf8(model_path);
    session_ = std::make_unique<Ort::Session>(env_, wpath.c_str(), opts);
    allocator_ = std::make_unique<Ort::AllocatorWithDefaultOptions>();

    state_.resize(size_state_, 0.f);
    context_.assign(context_samples_, 0.f);
    sr_ = static_cast<int64_t>(sample_rate_);

    input_dims_[0] = 1;
    input_dims_[1] = static_cast<int64_t>(effective_window_size_);
  }

  float get_speech_probability(float const* samples, size_t num_samples) {
    if (num_samples == 0)
      return 0.f;

    float max_prob = 0.f;
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeCPU);

    size_t offset = 0;
    while (offset + window_size_samples_ <= num_samples) {
      std::vector<float> new_data(effective_window_size_, 0.f);
      std::copy(context_.begin(), context_.end(), new_data.begin());
      std::copy(samples + offset, samples + offset + window_size_samples_,
                new_data.begin() + context_samples_);

      Ort::Value input_ort = Ort::Value::CreateTensor<float>(memory_info, new_data.data(),
                                                             new_data.size(), input_dims_, 2);
      Ort::Value state_ort = Ort::Value::CreateTensor<float>(memory_info, state_.data(),
                                                             state_.size(), state_dims_, 3);
      Ort::Value sr_ort = Ort::Value::CreateTensor<int64_t>(memory_info, &sr_, 1, sr_dims_, 1);

      const char* input_names[] = {"input", "state", "sr"};
      const char* output_names[] = {"output", "stateN"};
      std::array<Ort::Value, 3> inputs = {std::move(input_ort), std::move(state_ort),
                                          std::move(sr_ort)};

      auto outputs = session_->Run(Ort::RunOptions{nullptr}, input_names, inputs.data(),
                                   inputs.size(), output_names, 2);

      float prob = *outputs[0].GetTensorData<float>();
      max_prob = std::max(max_prob, prob);

      float const* stateN = outputs[1].GetTensorData<float>();
      std::memcpy(state_.data(), stateN, size_state_ * sizeof(float));

      std::copy(new_data.end() - static_cast<long>(context_samples_), new_data.end(),
                context_.begin());

      offset += window_size_samples_;
    }

    return max_prob;
  }

  void reset() {
    std::memset(state_.data(), 0, state_.size() * sizeof(float));
    std::fill(context_.begin(), context_.end(), 0.f);
  }

private:
  int sample_rate_;
  int window_size_samples_;
  int context_samples_;
  int effective_window_size_;
  static constexpr unsigned int size_state_ = 2 * 1 * 128;

  Ort::Env env_;
  std::unique_ptr<Ort::Session> session_;
  std::unique_ptr<Ort::AllocatorWithDefaultOptions> allocator_;
  std::vector<float> state_;
  std::vector<float> context_;
  int64_t sr_;
  int64_t input_dims_[2] = {1, 0};
  int64_t state_dims_[3] = {2, 1, 128};
  int64_t sr_dims_[1] = {1};
};

SileroVad::SileroVad(std::string const& model_path, int sample_rate)
    : impl_(std::make_unique<Impl>(model_path, sample_rate)) {}

SileroVad::~SileroVad() = default;

float SileroVad::get_speech_probability(float const* samples, size_t num_samples) {
  return impl_->get_speech_probability(samples, num_samples);
}

void SileroVad::reset() {
  impl_->reset();
}

}  // namespace ears

#endif  // EARS_HAS_ONNX
