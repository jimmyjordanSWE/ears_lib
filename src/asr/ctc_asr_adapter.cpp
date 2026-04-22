#if EARS_HAS_ONNX

#include "ears/asr/ctc_asr_adapter.hpp"

#include <cstdint>
#include <limits>
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

}  // namespace

class CtcAsrAdapter::Impl {
public:
  explicit Impl(std::string const& model_path, int sample_rate, std::string provider,
                std::string runtime)
      : sample_rate_(sample_rate),
        provider_request_(std::move(provider)),
        runtime_id_(asr_onnx::normalize_runtime_id(runtime)),
        env_(ORT_LOGGING_LEVEL_WARNING, "ears_ctc_asr") {
    model_file_ = asr_onnx::resolve_model_file(model_path, {"model.onnx", "ctc.onnx", "asr.onnx"});
    base_dir_ = asr_onnx::model_base_dir(model_path, model_file_);
    token_table_ = asr_onnx::load_token_table_with_parent_fallback(base_dir_);

    try {
      Ort::SessionOptions opts;
      asr_onnx::configure_session_options(opts);
      selected_provider_ =
          asr_onnx::append_onnx_provider_or_cpu(opts, runtime_id_, provider_request_);
      std::wstring wpath = asr_onnx::to_wide(model_file_);
      session_ = std::make_unique<Ort::Session>(env_, wpath.c_str(), opts);
      loaded_ = true;
      input_names_ = session_->GetConst().GetInputNames();
      output_names_ = session_->GetConst().GetOutputNames();
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
        {"adapter", "ctc"},
        {"runtime", runtime_id_},
        {"provider_request", provider_request_},
        {"provider_selected", selected_provider_},
        {"model_loaded", loaded_},
        {"model_file", model_file_},
        {"token_source", token_table_.source_path},
        {"token_count", token_table_.tokens.size()},
    };

    if (!loaded_ || !session_) {
      if (!last_error_.empty()) {
        meta["error"] = last_error_;
      }
      asr_onnx::set_adapter_fallback_result(r, "ctc", num_samples, 0.2f, "model_unavailable", meta);
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
      std::vector<const char*> input_name_ptrs;
      std::vector<Ort::Value> input_values;
      std::vector<std::vector<float>> float_storage;
      std::vector<std::vector<int64_t>> int64_storage;
      std::vector<std::vector<int32_t>> int32_storage;

      if (!prepare_inputs(samples, num_samples, input_name_ptrs, input_values, float_storage,
                          int64_storage, int32_storage, meta)) {
        asr_onnx::set_adapter_fallback_result(r, "ctc", num_samples, 0.3f, "input_bind_failed",
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
      asr_onnx::set_adapter_fallback_result(r, "ctc", num_samples, 0.2f, "runtime_error", meta);
      return r;
    } catch (...) {
      asr_onnx::set_adapter_fallback_result(r, "ctc", num_samples, 0.2f, "unknown_error", meta);
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

  DecodeResult decode_ctc_logits(Ort::Value const& value) const {
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

    auto choose_vocab_axis = [&](std::vector<int64_t> const& dims) -> size_t {
      size_t axis = dims.size() - 1;
      if (token_table_.tokens.empty()) {
        return axis;
      }
      int64_t target = static_cast<int64_t>(token_table_.tokens.size());
      int64_t best_distance = std::numeric_limits<int64_t>::max();
      for (size_t i = 0; i < dims.size(); ++i) {
        if (dims[i] <= 1) {
          continue;
        }
        int64_t distance = std::llabs(dims[i] - target);
        if (distance < best_distance) {
          best_distance = distance;
          axis = i;
        }
      }
      return axis;
    };

    size_t vocab_axis = choose_vocab_axis(shape);
    int64_t vocab = shape[vocab_axis];
    if (vocab <= 1) {
      return decoded;
    }

    float const* logits = value.GetTensorData<float>();
    std::vector<int64_t> ids;
    double conf_sum = 0.0;
    size_t conf_count = 0;
    int64_t prev = -1;
    int blank_id = token_table_.blank_id;
    if (blank_id < 0 || blank_id >= vocab) {
      blank_id = static_cast<int>(vocab - 1);
    }

    auto pick_best_from_accessor = [&](auto&& at, int64_t width) -> std::pair<int64_t, float> {
      int64_t best_id = 0;
      float best = at(0);
      std::vector<float> local_logits(static_cast<size_t>(width), 0.0f);
      local_logits[0] = best;
      for (int64_t v = 1; v < width; ++v) {
        float val = at(v);
        local_logits[static_cast<size_t>(v)] = val;
        if (val > best) {
          best = val;
          best_id = v;
        }
      }
      float conf = asr_onnx::softmax_probability(local_logits.data(), width, best_id);
      return {best_id, conf};
    };

    auto emit_id = [&](int64_t id, float conf) {
      if (id == blank_id) {
        prev = -1;
        return;
      }
      if (id == prev) {
        return;
      }
      prev = id;
      ids.push_back(id);
      conf_sum += conf;
      ++conf_count;
    };

    if (rank == 2) {
      if (vocab_axis == 1) {  // [T, V]
        int64_t rows = shape[0];
        ids.reserve(static_cast<size_t>(rows));
        for (int64_t t = 0; t < rows; ++t) {
          float const* row_ptr = logits + t * vocab;
          int64_t best_id = 0;
          float best = row_ptr[0];
          for (int64_t v = 1; v < vocab; ++v) {
            if (row_ptr[v] > best) {
              best = row_ptr[v];
              best_id = v;
            }
          }
          emit_id(best_id, asr_onnx::softmax_probability(row_ptr, vocab, best_id));
        }
      } else {  // [V, T]
        int64_t rows = shape[1];
        ids.reserve(static_cast<size_t>(rows));
        for (int64_t t = 0; t < rows; ++t) {
          auto at = [&](int64_t v) { return logits[v * rows + t]; };
          auto best = pick_best_from_accessor(at, vocab);
          emit_id(best.first, best.second);
        }
      }
    } else if (rank == 3 && shape[0] >= 1) {
      int64_t batch = shape[0];
      int64_t d1 = shape[1];
      int64_t d2 = shape[2];
      int64_t b = 0;  // benchmark path uses single utterance; decode first batch.
      (void)batch;
      if (vocab_axis == 2) {  // [B, T, V]
        int64_t rows = d1;
        ids.reserve(static_cast<size_t>(rows));
        for (int64_t t = 0; t < rows; ++t) {
          float const* row_ptr = logits + (b * d1 + t) * d2;
          int64_t best_id = 0;
          float best = row_ptr[0];
          for (int64_t v = 1; v < vocab; ++v) {
            if (row_ptr[v] > best) {
              best = row_ptr[v];
              best_id = v;
            }
          }
          emit_id(best_id, asr_onnx::softmax_probability(row_ptr, vocab, best_id));
        }
      } else if (vocab_axis == 1) {  // [B, V, T]
        int64_t rows = d2;
        ids.reserve(static_cast<size_t>(rows));
        for (int64_t t = 0; t < rows; ++t) {
          auto at = [&](int64_t v) { return logits[(b * d1 + v) * d2 + t]; };
          auto best = pick_best_from_accessor(at, vocab);
          emit_id(best.first, best.second);
        }
      } else {  // [V, T, B] or unusual
        int64_t rows = d1;
        ids.reserve(static_cast<size_t>(rows));
        for (int64_t t = 0; t < rows; ++t) {
          auto at = [&](int64_t v) { return logits[(v * d1 + t) * d2 + b]; };
          auto best = pick_best_from_accessor(at, vocab);
          emit_id(best.first, best.second);
        }
      }
    } else {
      // Fallback: flatten all non-vocab dimensions into time rows.
      int64_t rows = 1;
      for (size_t i = 0; i < rank; ++i) {
        if (i == vocab_axis) {
          continue;
        }
        rows *= shape[i];
      }
      if (rows <= 0) {
        return decoded;
      }
      ids.reserve(static_cast<size_t>(rows));
      for (int64_t row = 0; row < rows; ++row) {
        float const* ptr = logits + row * vocab;
        int64_t best_id = 0;
        float best = ptr[0];
        for (int64_t v = 1; v < vocab; ++v) {
          if (ptr[v] > best) {
            best = ptr[v];
            best_id = v;
          }
        }
        emit_id(best_id, asr_onnx::softmax_probability(ptr, vocab, best_id));
      }
    }

    decoded.text = asr_onnx::decode_ids_to_text(ids, token_table_.tokens, blank_id, false, true);
    if (decoded.text.empty() && !ids.empty()) {
      decoded.text = asr_onnx::format_token_id_preview("ctc", ids);
    }
    decoded.confidence =
        conf_count == 0 ? 0.0f : static_cast<float>(conf_sum / static_cast<double>(conf_count));
    decoded.token_count = ids.size();
    return decoded;
  }

  DecodeResult decode_outputs(std::vector<Ort::Value> const& outputs, nlohmann::json& meta) const {
    DecodeResult decoded;

    if (asr_onnx::decode_first_string_output(outputs, meta, decoded)) {
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
      decoded = decode_ctc_logits(outputs[logits_index]);
      meta["output_type"] = "float_logits";
      meta["logits_output_index"] = logits_index;
      meta["logits_vocab"] = best_vocab;
      if (!decoded.text.empty() || decoded.token_count > 0) {
        return decoded;
      }
    }

    if (asr_onnx::decode_first_token_id_output(outputs, token_table_.tokens, token_table_.blank_id,
                                               true, true, 0.75f, meta, decoded)) {
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
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  std::string model_file_;
  std::string base_dir_;
  asr_onnx::TokenTable token_table_;
  std::string last_error_;
  bool loaded_ = false;
};

CtcAsrAdapter::CtcAsrAdapter(std::string const& model_path, int sample_rate, std::string provider)
    : impl_(std::make_unique<Impl>(model_path, sample_rate, std::move(provider), "onnx_runtime")) {}

CtcAsrAdapter::~CtcAsrAdapter() = default;

AsrResult CtcAsrAdapter::recognize(float const* samples, size_t num_samples) {
  return impl_->recognize(samples, num_samples);
}

}  // namespace ears

#endif  // EARS_HAS_ONNX
