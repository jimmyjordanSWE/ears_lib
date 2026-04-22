#if EARS_HAS_ONNX

#include "ears/asr/rnnt_asr_adapter.hpp"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <nlohmann/json.hpp>
#include <onnxruntime_cxx_api.h>
#include <string>
#include <utility>
#include <vector>

#include "ears/mel.hpp"
#include "onnx_adapter_common.hpp"
#include "onnx_asr_common.hpp"
#include "onnx_audio_input_binding.hpp"
#include "onnx_decode_fallback.hpp"

namespace ears {

namespace {

struct DecodeResult {
  std::string text;
  float confidence = 0.0f;
  size_t token_count = 0;
};

std::string pick_existing_file(std::filesystem::path const& base,
                               std::vector<std::string> const& candidates) {
  for (auto const& candidate : candidates) {
    std::filesystem::path path = base / candidate;
    if (asr_onnx::fs_is_regular_file(path)) {
      return path.string();
    }
  }
  return "";
}

}  // namespace

class RnntAsrAdapter::Impl {
public:
  explicit Impl(std::string const& model_path, int sample_rate, std::string provider,
                std::string runtime)
      : env_(ORT_LOGGING_LEVEL_WARNING, "ears_rnnt_asr") {
    sample_rate_ = sample_rate;
    provider_request_ = std::move(provider);
    runtime_id_ = asr_onnx::normalize_runtime_id(runtime);
    std::filesystem::path input_path(model_path);
    std::filesystem::path model_root;
    if (asr_onnx::fs_is_directory(input_path)) {
      model_root = input_path;
    } else if (input_path.has_parent_path()) {
      model_root = input_path.parent_path();
    } else {
      model_root = ".";
    }

    std::string maybe_encoder =
        pick_existing_file(model_root, {"encoder-model.int8.onnx", "encoder_model.int8.onnx",
                                        "encoder-model.onnx", "encoder_model.onnx"});
    std::string maybe_decoder = pick_existing_file(
        model_root, {"decoder_joint-model.int8.onnx", "decoder_joint_model.int8.onnx",
                     "decoder_joint-model.onnx", "decoder_joint_model.onnx"});

    if (!maybe_encoder.empty() && !maybe_decoder.empty()) {
      split_encoder_file_ = std::move(maybe_encoder);
      split_decoder_file_ = std::move(maybe_decoder);
      model_file_ = split_encoder_file_;
      base_dir_ = model_root.string();
      split_rnnt_ = true;
    } else {
      model_file_ =
          asr_onnx::resolve_model_file(model_path, {"model.onnx", "rnnt.onnx", "transducer.onnx"});
      base_dir_ = asr_onnx::model_base_dir(model_path, model_file_);
    }

    token_table_ = asr_onnx::load_token_table_with_parent_fallback(base_dir_);

    try {
      Ort::SessionOptions opts;
      asr_onnx::configure_session_options(opts);
      selected_provider_ =
          asr_onnx::append_onnx_provider_or_cpu(opts, runtime_id_, provider_request_);
      if (split_rnnt_) {
        std::wstring encoder_w = asr_onnx::to_wide(split_encoder_file_);
        std::wstring decoder_w = asr_onnx::to_wide(split_decoder_file_);
        encoder_session_ = std::make_unique<Ort::Session>(env_, encoder_w.c_str(), opts);
        decoder_joint_session_ = std::make_unique<Ort::Session>(env_, decoder_w.c_str(), opts);
        encoder_input_names_ = encoder_session_->GetConst().GetInputNames();
        encoder_output_names_ = encoder_session_->GetConst().GetOutputNames();
        decoder_input_names_ = decoder_joint_session_->GetConst().GetInputNames();
        decoder_output_names_ = decoder_joint_session_->GetConst().GetOutputNames();
      } else {
        std::wstring wpath = asr_onnx::to_wide(model_file_);
        session_ = std::make_unique<Ort::Session>(env_, wpath.c_str(), opts);
        input_names_ = session_->GetConst().GetInputNames();
        output_names_ = session_->GetConst().GetOutputNames();
      }
      loaded_ = true;
    } catch (Ort::Exception const& e) {
      last_error_ = e.what();
      loaded_ = false;
    } catch (...) {
      last_error_ = "session_init_failed";
      loaded_ = false;
    }
  }

  AsrResult recognize(float const* samples, size_t num_samples) {
    AsrResult r;
    nlohmann::json meta = {
        {"adapter", "rnnt"},
        {"runtime", runtime_id_},
        {"provider_request", provider_request_},
        {"provider_selected", selected_provider_},
        {"model_loaded", loaded_},
        {"model_file", model_file_},
        {"split_rnnt", split_rnnt_},
        {"split_encoder", split_encoder_file_},
        {"split_decoder_joint", split_decoder_file_},
        {"token_source", token_table_.source_path},
        {"token_count", token_table_.tokens.size()},
    };

    bool const split_sessions_ready = split_rnnt_ && encoder_session_ && decoder_joint_session_;
    if (!loaded_ || (!session_ && !split_sessions_ready)) {
      if (!last_error_.empty()) {
        meta["error"] = last_error_;
      }
      asr_onnx::set_adapter_fallback_result(r, "rnnt", num_samples, 0.2f, "model_unavailable",
                                            meta);
      return r;
    }

    if (samples == nullptr || num_samples == 0) {
      r.text = "";
      r.confidence = 0.0f;
      meta["decode"] = "empty_input";
      r.json = meta.dump();
      return r;
    }

    try {
      if (split_rnnt_) {
        DecodeResult decoded = decode_split_rnnt(samples, num_samples, meta);
        asr_onnx::set_decoded_result(r, std::move(decoded.text), decoded.confidence,
                                     decoded.token_count, meta);
        return r;
      }

      std::vector<const char*> input_name_ptrs;
      std::vector<Ort::Value> input_values;
      std::vector<std::vector<float>> float_storage;
      std::vector<std::vector<int64_t>> int64_storage;
      std::vector<std::vector<int32_t>> int32_storage;

      if (!prepare_inputs(samples, num_samples, input_name_ptrs, input_values, float_storage,
                          int64_storage, int32_storage, meta)) {
        asr_onnx::set_adapter_fallback_result(r, "rnnt", num_samples, 0.3f, "input_bind_failed",
                                              meta);
        return r;
      }

      std::vector<const char*> output_name_ptrs = asr_onnx::to_name_ptrs(output_names_);

      auto outputs =
          session_->Run(Ort::RunOptions{nullptr}, input_name_ptrs.data(), input_values.data(),
                        input_values.size(), output_name_ptrs.data(), output_name_ptrs.size());

      DecodeResult decoded = decode_outputs(outputs, meta);
      asr_onnx::set_decoded_result(r, std::move(decoded.text), decoded.confidence,
                                   decoded.token_count, meta);
      return r;
    } catch (Ort::Exception const& e) {
      meta["error"] = e.what();
      asr_onnx::set_adapter_fallback_result(r, "rnnt", num_samples, 0.2f, "runtime_error", meta);
      return r;
    } catch (...) {
      asr_onnx::set_adapter_fallback_result(r, "rnnt", num_samples, 0.2f, "unknown_error", meta);
      return r;
    }
  }

private:
  bool prepare_inputs(float const* samples, size_t num_samples,
                      std::vector<const char*>& input_name_ptrs,
                      std::vector<Ort::Value>& input_values,
                      std::vector<std::vector<float>>& float_storage,
                      std::vector<std::vector<int64_t>>& int64_storage,
                      std::vector<std::vector<int32_t>>& int32_storage,
                      nlohmann::json& meta) const {
    if (!session_) {
      return false;
    }
    return asr_onnx::prepare_audio_inputs(*session_, input_names_, sample_rate_, samples,
                                          num_samples, input_name_ptrs, input_values, float_storage,
                                          int64_storage, int32_storage, meta);
  }

  DecodeResult decode_rnnt_logits(Ort::Value const& value) const {
    DecodeResult decoded;
    if (!value.IsTensor()) {
      return decoded;
    }
    auto info = value.GetTensorTypeAndShapeInfo();
    if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      return decoded;
    }

    std::vector<int64_t> shape = asr_onnx::shape_with_dynamic_as_one(info.GetShape());
    size_t const rank = shape.size();
    if (rank < 2) {
      return decoded;
    }

    int64_t vocab = shape.back();
    if (vocab <= 1) {
      return decoded;
    }

    float const* logits = value.GetTensorData<float>();
    std::vector<int64_t> ids;
    double conf_sum = 0.0;
    size_t conf_count = 0;
    int blank_id = token_table_.blank_id;

    auto choose_token = [&](float const* row) {
      int64_t best_id = 0;
      float best = row[0];
      for (int64_t v = 1; v < vocab; ++v) {
        if (row[v] > best) {
          best = row[v];
          best_id = v;
        }
      }
      return best_id;
    };

    if (rank == 4) {
      int64_t t_dim = shape[1];
      int64_t u_dim = shape[2];
      for (int64_t t = 0; t < t_dim; ++t) {
        for (int64_t u = 0; u < u_dim; ++u) {
          int64_t offset = (t * u_dim + u) * vocab;
          float const* row = logits + offset;
          int64_t id = choose_token(row);
          if (id == blank_id) {
            break;
          }
          ids.push_back(id);
          conf_sum += asr_onnx::softmax_probability(row, vocab, id);
          ++conf_count;
        }
      }
    } else if (rank == 3) {
      int64_t d0 = shape[0];
      int64_t d1 = shape[1];
      if (d0 == 1 || d1 == 1) {
        int64_t rows = std::max(d0, d1);
        for (int64_t r = 0; r < rows; ++r) {
          int64_t offset = r * vocab;
          float const* row = logits + offset;
          int64_t id = choose_token(row);
          if (id == blank_id) {
            continue;
          }
          ids.push_back(id);
          conf_sum += asr_onnx::softmax_probability(row, vocab, id);
          ++conf_count;
        }
      } else {
        for (int64_t t = 0; t < d0; ++t) {
          for (int64_t u = 0; u < d1; ++u) {
            int64_t offset = (t * d1 + u) * vocab;
            float const* row = logits + offset;
            int64_t id = choose_token(row);
            if (id == blank_id) {
              break;
            }
            ids.push_back(id);
            conf_sum += asr_onnx::softmax_probability(row, vocab, id);
            ++conf_count;
          }
        }
      }
    } else {
      int64_t rows = shape[0];
      for (int64_t r = 0; r < rows; ++r) {
        int64_t offset = r * vocab;
        float const* row = logits + offset;
        int64_t id = choose_token(row);
        if (id == blank_id) {
          continue;
        }
        ids.push_back(id);
        conf_sum += asr_onnx::softmax_probability(row, vocab, id);
        ++conf_count;
      }
    }

    decoded.text = asr_onnx::decode_ids_to_text(ids, token_table_.tokens, blank_id, false, true);
    if (decoded.text.empty() && !ids.empty()) {
      decoded.text = asr_onnx::format_token_id_preview("rnnt", ids);
    }
    decoded.confidence =
        conf_count == 0 ? 0.0f : static_cast<float>(conf_sum / static_cast<double>(conf_count));
    decoded.token_count = ids.size();
    return decoded;
  }

  DecodeResult decode_split_rnnt(float const* samples, size_t num_samples,
                                 nlohmann::json& meta) const {
    DecodeResult decoded;
    if (!encoder_session_ || !decoder_joint_session_) {
      return decoded;
    }

    constexpr int64_t kFeatureSize = 80;
    std::vector<float> mel_time_major = compute_log_mel(samples, num_samples, kFeatureSize);
    int64_t frames = static_cast<int64_t>(mel_time_major.size() / kFeatureSize);
    if (frames <= 0) {
      return decoded;
    }

    std::vector<float> mel_channel_major(static_cast<size_t>(kFeatureSize * frames), 0.0f);
    for (int64_t t = 0; t < frames; ++t) {
      for (int64_t f = 0; f < kFeatureSize; ++f) {
        mel_channel_major[static_cast<size_t>(f * frames + t)] =
            mel_time_major[static_cast<size_t>(t * kFeatureSize + f)];
      }
    }

    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<int64_t> encoder_shape = {1, kFeatureSize, frames};
    std::vector<int64_t> encoder_length = {frames};
    std::vector<const char*> encoder_input_names;
    std::vector<Ort::Value> encoder_inputs;
    encoder_input_names.reserve(encoder_input_names_.size());
    encoder_inputs.reserve(encoder_input_names_.size());

    for (size_t i = 0; i < encoder_input_names_.size(); ++i) {
      std::string const& raw_name = encoder_input_names_[i];
      std::string const lowered = asr_onnx::to_lower_ascii(raw_name);
      encoder_input_names.push_back(raw_name.c_str());

      auto info = encoder_session_->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo();
      auto elem_type = info.GetElementType();
      if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
          (asr_onnx::has_any_substring(lowered, {"audio", "signal", "feat", "mel"}) ||
           info.GetShape().size() >= 3)) {
        encoder_inputs.emplace_back(
            Ort::Value::CreateTensor<float>(mem, mel_channel_major.data(), mel_channel_major.size(),
                                            encoder_shape.data(), encoder_shape.size()));
      } else if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
        std::vector<int64_t> len_shape = {1};
        encoder_inputs.emplace_back(Ort::Value::CreateTensor<int64_t>(
            mem, encoder_length.data(), encoder_length.size(), len_shape.data(), len_shape.size()));
      } else {
        return decoded;
      }
    }

    std::vector<const char*> encoder_output_names;
    encoder_output_names.reserve(encoder_output_names_.size());
    for (auto const& name : encoder_output_names_) {
      encoder_output_names.push_back(name.c_str());
    }

    auto encoder_outputs = encoder_session_->Run(
        Ort::RunOptions{nullptr}, encoder_input_names.data(), encoder_inputs.data(),
        encoder_inputs.size(), encoder_output_names.data(), encoder_output_names.size());

    Ort::Value const* encoded_value = nullptr;
    Ort::Value const* encoded_len_value = nullptr;
    for (size_t i = 0; i < encoder_outputs.size(); ++i) {
      auto const& output = encoder_outputs[i];
      if (!output.IsTensor()) {
        continue;
      }
      auto info = output.GetTensorTypeAndShapeInfo();
      if (info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT && !encoded_value) {
        encoded_value = &output;
      } else if ((info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 ||
                  info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) &&
                 !encoded_len_value) {
        encoded_len_value = &output;
      }
    }
    if (!encoded_value) {
      return decoded;
    }

    auto enc_info = encoded_value->GetTensorTypeAndShapeInfo();
    std::vector<int64_t> enc_shape = asr_onnx::shape_with_dynamic_as_one(enc_info.GetShape());
    if (enc_shape.size() < 3) {
      return decoded;
    }
    int64_t enc_hidden = enc_shape[1];
    int64_t enc_time = enc_shape[2];
    bool enc_layout_bdt = true;
    if (enc_shape.size() >= 3 && enc_shape[2] == 1024 && enc_shape[1] != 1024) {
      enc_hidden = enc_shape[2];
      enc_time = enc_shape[1];
      enc_layout_bdt = false;  // [B,T,D]
    }

    int64_t decode_steps = enc_time;
    if (encoded_len_value) {
      std::vector<int64_t> lengths = asr_onnx::extract_int_tensor_ids(*encoded_len_value);
      if (!lengths.empty()) {
        decode_steps = std::max<int64_t>(1, std::min<int64_t>(enc_time, lengths[0]));
      }
    }

    float const* enc_data = encoded_value->GetTensorData<float>();
    std::vector<int64_t> state_shape = {2, 1, 640};
    for (size_t i = 0; i < decoder_input_names_.size(); ++i) {
      std::string const lowered = asr_onnx::to_lower_ascii(decoder_input_names_[i]);
      if (!asr_onnx::has_any_substring(lowered, {"input_states_1"})) {
        continue;
      }
      auto state_info = decoder_joint_session_->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo();
      std::vector<int64_t> candidate = asr_onnx::shape_with_dynamic_as_one(state_info.GetShape());
      if (candidate.size() == 3 && asr_onnx::element_count(candidate) > 0) {
        state_shape = std::move(candidate);
      }
      break;
    }
    int64_t state_count = asr_onnx::element_count(state_shape);
    if (state_count <= 0) {
      state_shape = {2, 1, 640};
      state_count = asr_onnx::element_count(state_shape);
    }
    std::vector<float> state1(static_cast<size_t>(state_count), 0.0f);
    std::vector<float> state2(static_cast<size_t>(state_count), 0.0f);
    int blank_id = token_table_.blank_id;
    if (blank_id < 0 || blank_id >= static_cast<int>(token_table_.tokens.size())) {
      if (!token_table_.tokens.empty()) {
        blank_id = static_cast<int>(token_table_.tokens.size() - 1);
      } else {
        blank_id = 1024;
      }
    }
    int64_t current_target = static_cast<int64_t>(blank_id);

    std::vector<int64_t> emitted_ids;
    emitted_ids.reserve(static_cast<size_t>(decode_steps));
    double conf_sum = 0.0;
    size_t conf_count = 0;
    constexpr int kMaxSymbolsPerFrame = 4;
    constexpr size_t kMaxTokens = 256;

    for (int64_t t = 0; t < decode_steps && emitted_ids.size() < kMaxTokens; ++t) {
      std::vector<float> encoder_step(static_cast<size_t>(enc_hidden), 0.0f);
      if (enc_layout_bdt) {
        for (int64_t d = 0; d < enc_hidden; ++d) {
          encoder_step[static_cast<size_t>(d)] = enc_data[static_cast<size_t>(d * enc_time + t)];
        }
      } else {
        for (int64_t d = 0; d < enc_hidden; ++d) {
          encoder_step[static_cast<size_t>(d)] = enc_data[static_cast<size_t>(t * enc_hidden + d)];
        }
      }

      for (int iter = 0; iter < kMaxSymbolsPerFrame && emitted_ids.size() < kMaxTokens; ++iter) {
        std::vector<const char*> decoder_input_names;
        std::vector<Ort::Value> decoder_inputs;
        decoder_input_names.reserve(decoder_input_names_.size());
        decoder_inputs.reserve(decoder_input_names_.size());

        std::vector<int64_t> encoder_step_shape = {1, enc_hidden, 1};
        std::vector<int64_t> target_shape = {1, 1};
        std::vector<int64_t> target_len_shape = {1};
        std::vector<int32_t> target_i32_storage = {static_cast<int32_t>(current_target)};
        std::vector<int64_t> target_i64_storage = {current_target};
        std::vector<int32_t> target_len_i32_storage = {1};
        std::vector<int64_t> target_len_i64_storage = {1};

        for (size_t i = 0; i < decoder_input_names_.size(); ++i) {
          std::string const& raw_name = decoder_input_names_[i];
          std::string const lowered = asr_onnx::to_lower_ascii(raw_name);
          auto input_info = decoder_joint_session_->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo();
          auto input_elem_type = input_info.GetElementType();
          decoder_input_names.push_back(raw_name.c_str());
          if (asr_onnx::has_any_substring(lowered, {"encoder_outputs", "encoder"})) {
            decoder_inputs.emplace_back(Ort::Value::CreateTensor<float>(
                mem, encoder_step.data(), encoder_step.size(), encoder_step_shape.data(),
                encoder_step_shape.size()));
            continue;
          }
          if (asr_onnx::has_any_substring(lowered, {"targets"}) &&
              !asr_onnx::has_any_substring(lowered, {"length"})) {
            if (input_elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
              decoder_inputs.emplace_back(Ort::Value::CreateTensor<int64_t>(
                  mem, target_i64_storage.data(), target_i64_storage.size(), target_shape.data(),
                  target_shape.size()));
            } else {
              decoder_inputs.emplace_back(Ort::Value::CreateTensor<int32_t>(
                  mem, target_i32_storage.data(), target_i32_storage.size(), target_shape.data(),
                  target_shape.size()));
            }
            continue;
          }
          if (asr_onnx::has_any_substring(lowered, {"target_length", "prednet_lengths"})) {
            if (input_elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
              decoder_inputs.emplace_back(Ort::Value::CreateTensor<int64_t>(
                  mem, target_len_i64_storage.data(), target_len_i64_storage.size(),
                  target_len_shape.data(), target_len_shape.size()));
            } else {
              decoder_inputs.emplace_back(Ort::Value::CreateTensor<int32_t>(
                  mem, target_len_i32_storage.data(), target_len_i32_storage.size(),
                  target_len_shape.data(), target_len_shape.size()));
            }
            continue;
          }
          if (asr_onnx::has_any_substring(lowered, {"input_states_1"})) {
            decoder_inputs.emplace_back(Ort::Value::CreateTensor<float>(
                mem, state1.data(), state1.size(), state_shape.data(), state_shape.size()));
            continue;
          }
          if (asr_onnx::has_any_substring(lowered, {"input_states_2"})) {
            decoder_inputs.emplace_back(Ort::Value::CreateTensor<float>(
                mem, state2.data(), state2.size(), state_shape.data(), state_shape.size()));
            continue;
          }
          return decoded;
        }

        std::vector<const char*> decoder_output_names;
        decoder_output_names.reserve(decoder_output_names_.size());
        for (auto const& name : decoder_output_names_) {
          decoder_output_names.push_back(name.c_str());
        }

        auto decoder_outputs = decoder_joint_session_->Run(
            Ort::RunOptions{nullptr}, decoder_input_names.data(), decoder_inputs.data(),
            decoder_inputs.size(), decoder_output_names.data(), decoder_output_names.size());

        Ort::Value const* logits_value = nullptr;
        for (size_t i = 0; i < decoder_outputs.size(); ++i) {
          auto const& output = decoder_outputs[i];
          if (!output.IsTensor()) {
            continue;
          }
          auto info = output.GetTensorTypeAndShapeInfo();
          if (info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            if (!logits_value &&
                asr_onnx::has_any_substring(asr_onnx::to_lower_ascii(decoder_output_names_[i]),
                                            {"outputs", "logit"})) {
              logits_value = &output;
            }
            if (asr_onnx::has_any_substring(asr_onnx::to_lower_ascii(decoder_output_names_[i]),
                                            {"output_states_1"})) {
              auto state_info = output.GetTensorTypeAndShapeInfo();
              std::vector<int64_t> shape =
                  asr_onnx::shape_with_dynamic_as_one(state_info.GetShape());
              state_shape = shape;
              size_t count = static_cast<size_t>(state_info.GetElementCount());
              state1.assign(output.GetTensorData<float>(), output.GetTensorData<float>() + count);
            } else if (asr_onnx::has_any_substring(
                           asr_onnx::to_lower_ascii(decoder_output_names_[i]),
                           {"output_states_2"})) {
              size_t count =
                  static_cast<size_t>(output.GetTensorTypeAndShapeInfo().GetElementCount());
              state2.assign(output.GetTensorData<float>(), output.GetTensorData<float>() + count);
            }
          }
        }
        if (!logits_value) {
          return decoded;
        }

        auto logits_info = logits_value->GetTensorTypeAndShapeInfo();
        std::vector<int64_t> logits_shape =
            asr_onnx::shape_with_dynamic_as_one(logits_info.GetShape());
        if (logits_shape.size() < 2) {
          return decoded;
        }
        int64_t vocab = logits_shape.back();
        int64_t rows = static_cast<int64_t>(logits_info.GetElementCount() / vocab);
        if (rows <= 0) {
          return decoded;
        }
        int64_t row_index = rows - 1;
        float const* logits = logits_value->GetTensorData<float>();
        float const* row = logits + row_index * vocab;
        if (blank_id < 0 || blank_id >= vocab) {
          blank_id = static_cast<int>(vocab - 1);
          if (current_target >= vocab) {
            current_target = blank_id;
          }
        }
        int64_t best_id = 0;
        float best_logit = row[0];
        for (int64_t v = 1; v < vocab; ++v) {
          if (row[v] > best_logit) {
            best_logit = row[v];
            best_id = v;
          }
        }

        if (best_id == blank_id) {
          break;
        }

        emitted_ids.push_back(best_id);
        conf_sum += asr_onnx::softmax_probability(row, vocab, best_id);
        ++conf_count;
        current_target = best_id;
      }
    }

    decoded.text =
        asr_onnx::decode_ids_to_text(emitted_ids, token_table_.tokens, blank_id, false, true);
    if (decoded.text.empty() && !emitted_ids.empty()) {
      decoded.text = asr_onnx::format_token_id_preview("rnnt", emitted_ids);
    }
    decoded.token_count = emitted_ids.size();
    decoded.confidence =
        conf_count == 0 ? 0.0f : static_cast<float>(conf_sum / static_cast<double>(conf_count));
    meta["split_blank_id"] = blank_id;
    meta["split_state_shape"] = state_shape;
    meta["split_decode_steps"] = decode_steps;
    meta["split_decode_tokens"] = decoded.token_count;
    return decoded;
  }

  DecodeResult decode_outputs(std::vector<Ort::Value> const& outputs, nlohmann::json& meta) const {
    DecodeResult decoded;

    if (asr_onnx::decode_first_string_output(outputs, meta, decoded)) {
      return decoded;
    }

    if (asr_onnx::decode_first_token_id_output(outputs, token_table_.tokens, token_table_.blank_id,
                                               false, true, 0.8f, meta, decoded)) {
      return decoded;
    }

    size_t logits_index = outputs.size();
    int64_t best_vocab = -1;
    for (size_t i = 0; i < outputs.size(); ++i) {
      Ort::Value const& output = outputs[i];
      if (!output.IsTensor()) {
        continue;
      }
      auto info = output.GetTensorTypeAndShapeInfo();
      if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        continue;
      }
      std::vector<int64_t> shape = asr_onnx::shape_with_dynamic_as_one(info.GetShape());
      if (shape.size() < 2) {
        continue;
      }
      int64_t vocab = shape.back();
      if (vocab > best_vocab) {
        best_vocab = vocab;
        logits_index = i;
      }
    }

    if (logits_index < outputs.size()) {
      decoded = decode_rnnt_logits(outputs[logits_index]);
      meta["output_type"] = "float_logits";
      meta["logits_output_index"] = logits_index;
      meta["logits_vocab"] = best_vocab;
      return decoded;
    }

    return decoded;
  }

private:
  int sample_rate_ = 16000;
  std::string provider_request_ = "auto";
  std::string selected_provider_ = "cpu";
  std::string runtime_id_ = "onnx_runtime";
  Ort::Env env_;
  std::unique_ptr<Ort::Session> session_;
  std::unique_ptr<Ort::Session> encoder_session_;
  std::unique_ptr<Ort::Session> decoder_joint_session_;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  std::vector<std::string> encoder_input_names_;
  std::vector<std::string> encoder_output_names_;
  std::vector<std::string> decoder_input_names_;
  std::vector<std::string> decoder_output_names_;
  std::string model_file_;
  std::string split_encoder_file_;
  std::string split_decoder_file_;
  std::string base_dir_;
  asr_onnx::TokenTable token_table_;
  std::string last_error_;
  bool loaded_ = false;
  bool split_rnnt_ = false;
};

RnntAsrAdapter::RnntAsrAdapter(std::string const& model_path, int sample_rate, std::string provider)
    : impl_(std::make_unique<Impl>(model_path, sample_rate, std::move(provider), "onnx_runtime")) {}

RnntAsrAdapter::~RnntAsrAdapter() = default;

AsrResult RnntAsrAdapter::recognize(float const* samples, size_t num_samples) {
  return impl_->recognize(samples, num_samples);
}

}  // namespace ears

#endif  // EARS_HAS_ONNX
