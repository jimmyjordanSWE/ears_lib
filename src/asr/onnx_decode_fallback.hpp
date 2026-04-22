#pragma once

#include <nlohmann/json.hpp>
#include <onnxruntime_cxx_api.h>
#include <string>
#include <vector>

#include "onnx_asr_common.hpp"

namespace ears::asr_onnx {

template <typename DecodeResultT>
inline bool decode_first_string_output(std::vector<Ort::Value> const& outputs, nlohmann::json& meta,
                                       DecodeResultT& decoded) {
  for (auto const& output : outputs) {
    std::string text = extract_first_string_tensor(output);
    if (text.empty()) {
      continue;
    }
    decoded.text = normalize_whitespace(text);
    decoded.confidence = 0.95f;
    decoded.token_count = decoded.text.empty() ? 0 : 1;
    meta["output_type"] = "string";
    return true;
  }
  return false;
}

template <typename DecodeResultT>
inline bool decode_first_token_id_output(std::vector<Ort::Value> const& outputs,
                                         std::vector<std::string> const& tokens, int blank_id,
                                         bool collapse_repeats, bool reset_repeat_on_blank,
                                         float non_empty_confidence, nlohmann::json& meta,
                                         DecodeResultT& decoded) {
  for (auto const& output : outputs) {
    std::vector<int64_t> ids = extract_int_tensor_ids(output);
    if (ids.empty()) {
      continue;
    }
    decoded.text =
        decode_ids_to_text(ids, tokens, blank_id, collapse_repeats, reset_repeat_on_blank);
    decoded.token_count = ids.size();
    decoded.confidence = decoded.text.empty() ? 0.0f : non_empty_confidence;
    meta["output_type"] = "token_ids";
    return !decoded.text.empty();
  }
  return false;
}

}  // namespace ears::asr_onnx
