#pragma once

#include <cstddef>
#include <cstdint>
#include <nlohmann/json.hpp>
#include <onnxruntime_cxx_api.h>
#include <string>
#include <vector>

#include "ears/mel.hpp"
#include "onnx_asr_common.hpp"

namespace ears::asr_onnx {

inline bool needs_feature_tensor(std::string const& input_name, size_t rank) {
  if (rank >= 3) {
    return true;
  }
  return has_any_substring(input_name, {"feat", "fbank", "mel", "spec"});
}

inline bool is_length_input(std::string const& input_name) {
  return has_any_substring(input_name, {"length", "len", "seq_len", "num_samples"});
}

inline bool is_sample_rate_input(std::string const& input_name) {
  return has_any_substring(input_name, {"sample_rate", "sampling_rate"});
}

inline bool is_attention_mask_input(std::string const& input_name) {
  return has_any_substring(input_name, {"attention_mask", "mask"});
}

inline bool prepare_audio_inputs(
    Ort::Session const& session, std::vector<std::string> const& input_names, int sample_rate,
    float const* samples, size_t num_samples, std::vector<const char*>& input_name_ptrs,
    std::vector<Ort::Value>& input_values, std::vector<std::vector<float>>& float_storage,
    std::vector<std::vector<int64_t>>& int64_storage,
    std::vector<std::vector<int32_t>>& int32_storage, nlohmann::json& meta) {
  if (samples == nullptr || num_samples == 0) {
    return false;
  }

  Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  std::vector<float> const waveform(samples, samples + num_samples);
  int64_t const waveform_len = static_cast<int64_t>(num_samples);
  constexpr int64_t kFeatureSize = 80;

  std::vector<float> mel_time_major;
  bool mel_ready = false;
  int64_t feature_frames = waveform_len;

  bool model_needs_features = false;
  for (size_t i = 0; i < input_names.size(); ++i) {
    std::string const name = to_lower_ascii(input_names[i]);
    auto tensor_info = session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo();
    if (tensor_info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
        needs_feature_tensor(name, tensor_info.GetShape().size())) {
      model_needs_features = true;
      break;
    }
  }
  if (model_needs_features) {
    mel_time_major = compute_log_mel(samples, num_samples, static_cast<int>(kFeatureSize));
    mel_ready = true;
    feature_frames = static_cast<int64_t>(mel_time_major.size() / kFeatureSize);
    if (feature_frames <= 0) {
      return false;
    }
  }

  input_name_ptrs.clear();
  input_values.clear();
  input_name_ptrs.reserve(input_names.size());
  input_values.reserve(input_names.size());

  for (size_t i = 0; i < input_names.size(); ++i) {
    std::string const& raw_name = input_names[i];
    std::string const name = to_lower_ascii(raw_name);
    auto tensor_info = session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo();
    auto elem_type = tensor_info.GetElementType();
    std::vector<int64_t> shape = tensor_info.GetShape();
    size_t const rank = shape.size();
    input_name_ptrs.push_back(raw_name.c_str());

    if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      if (needs_feature_tensor(name, rank)) {
        if (!mel_ready) {
          mel_time_major = compute_log_mel(samples, num_samples, static_cast<int>(kFeatureSize));
          mel_ready = true;
        }
        int64_t frames = static_cast<int64_t>(mel_time_major.size() / kFeatureSize);
        if (frames <= 0) {
          return false;
        }

        bool time_major = true;
        if (rank >= 3) {
          if (shape[1] == kFeatureSize) {
            time_major = false;
          } else if (shape[2] == kFeatureSize) {
            time_major = true;
          }
        } else if (rank == 2 && shape[0] == kFeatureSize) {
          time_major = false;
        }

        float_storage.emplace_back();
        std::vector<float>& storage = float_storage.back();
        std::vector<int64_t> mel_shape;
        if (time_major) {
          mel_shape = (rank <= 2) ? std::vector<int64_t>{frames, kFeatureSize}
                                  : std::vector<int64_t>{1, frames, kFeatureSize};
          storage = mel_time_major;
        } else {
          mel_shape = (rank <= 2) ? std::vector<int64_t>{kFeatureSize, frames}
                                  : std::vector<int64_t>{1, kFeatureSize, frames};
          storage.assign(static_cast<size_t>(frames * kFeatureSize), 0.0f);
          for (int64_t t = 0; t < frames; ++t) {
            for (int64_t f = 0; f < kFeatureSize; ++f) {
              storage[static_cast<size_t>(f * frames + t)] =
                  mel_time_major[static_cast<size_t>(t * kFeatureSize + f)];
            }
          }
        }

        input_values.emplace_back(Ort::Value::CreateTensor<float>(
            mem, storage.data(), storage.size(), mel_shape.data(), mel_shape.size()));
        meta["input_" + raw_name] = time_major ? "mel_time_major" : "mel_channel_major";
        continue;
      }

      if (is_sample_rate_input(name) && rank <= 1) {
        float_storage.emplace_back(1, static_cast<float>(sample_rate));
        std::vector<float>& storage = float_storage.back();
        std::vector<int64_t> sr_shape = {1};
        input_values.emplace_back(Ort::Value::CreateTensor<float>(
            mem, storage.data(), storage.size(), sr_shape.data(), sr_shape.size()));
        meta["input_" + raw_name] = "sample_rate";
        continue;
      }

      float_storage.push_back(waveform);
      std::vector<float>& storage = float_storage.back();
      std::vector<int64_t> wav_shape;
      if (rank <= 1) {
        wav_shape = {waveform_len};
      } else if (rank == 2) {
        wav_shape = {1, waveform_len};
      } else {
        wav_shape = {1, 1, waveform_len};
      }
      input_values.emplace_back(Ort::Value::CreateTensor<float>(
          mem, storage.data(), storage.size(), wav_shape.data(), wav_shape.size()));
      meta["input_" + raw_name] = "waveform";
      continue;
    }

    auto build_i64 = [&](std::vector<int64_t> values, std::vector<int64_t> tensor_shape) {
      int64_storage.emplace_back(std::move(values));
      std::vector<int64_t>& storage = int64_storage.back();
      input_values.emplace_back(Ort::Value::CreateTensor<int64_t>(
          mem, storage.data(), storage.size(), tensor_shape.data(), tensor_shape.size()));
    };

    auto build_i32 = [&](std::vector<int32_t> values, std::vector<int64_t> tensor_shape) {
      int32_storage.emplace_back(std::move(values));
      std::vector<int32_t>& storage = int32_storage.back();
      input_values.emplace_back(Ort::Value::CreateTensor<int32_t>(
          mem, storage.data(), storage.size(), tensor_shape.data(), tensor_shape.size()));
    };

    if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 ||
        elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
      if (is_length_input(name)) {
        int64_t effective_len = mel_ready ? feature_frames : waveform_len;
        std::vector<int64_t> tensor_shape =
            (rank <= 1) ? std::vector<int64_t>{1} : std::vector<int64_t>{1, 1};
        if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
          build_i64({effective_len}, tensor_shape);
        } else {
          build_i32({static_cast<int32_t>(effective_len)}, tensor_shape);
        }
        meta["input_" + raw_name] = "length";
        continue;
      }

      if (is_sample_rate_input(name)) {
        std::vector<int64_t> tensor_shape = {1};
        if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
          build_i64({static_cast<int64_t>(sample_rate)}, tensor_shape);
        } else {
          build_i32({sample_rate}, tensor_shape);
        }
        meta["input_" + raw_name] = "sample_rate";
        continue;
      }

      if (is_attention_mask_input(name)) {
        int64_t effective_len = mel_ready ? feature_frames : waveform_len;
        std::vector<int64_t> tensor_shape = (rank <= 1) ? std::vector<int64_t>{effective_len}
                                                        : std::vector<int64_t>{1, effective_len};
        if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
          build_i64(std::vector<int64_t>(static_cast<size_t>(effective_len), 1), tensor_shape);
        } else {
          build_i32(std::vector<int32_t>(static_cast<size_t>(effective_len), 1), tensor_shape);
        }
        meta["input_" + raw_name] = "attention_mask";
        continue;
      }

      std::vector<int64_t> fallback_shape = shape_with_dynamic_as_one(shape);
      int64_t count = element_count(fallback_shape);
      if (count <= 0) {
        return false;
      }
      if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
        build_i64(std::vector<int64_t>(static_cast<size_t>(count), 0), fallback_shape);
      } else {
        build_i32(std::vector<int32_t>(static_cast<size_t>(count), 0), fallback_shape);
      }
      meta["input_" + raw_name] = "fallback_int";
      continue;
    }

    return false;
  }

  return true;
}

}  // namespace ears::asr_onnx
