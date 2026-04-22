#pragma once

#include <algorithm>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <onnxruntime_cxx_api.h>
#include <string>
#include <utility>
#include <vector>

#include "ears/asr.hpp"
#include "onnx_asr_common.hpp"

namespace ears::asr_onnx {

inline void configure_session_options(Ort::SessionOptions& opts) {
  opts.SetIntraOpNumThreads(1);
  opts.SetInterOpNumThreads(1);
  opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
}

inline TokenTable load_token_table_with_parent_fallback(std::string const& base_dir) {
  TokenTable table = load_token_table(base_dir);
  if (!table.tokens.empty()) {
    return table;
  }
  std::filesystem::path parent = std::filesystem::path(base_dir).parent_path();
  if (parent.empty()) {
    return table;
  }
  return load_token_table(parent.string());
}

inline std::vector<const char*> to_name_ptrs(std::vector<std::string> const& names) {
  std::vector<const char*> ptrs;
  ptrs.reserve(names.size());
  for (auto const& name : names) {
    ptrs.push_back(name.c_str());
  }
  return ptrs;
}

inline void set_adapter_fallback_result(AsrResult& out, std::string const& adapter_name,
                                        size_t num_samples, float confidence,
                                        std::string const& decode_reason, nlohmann::json& meta) {
  out.text = "[" + adapter_name + ": " + std::to_string(num_samples) + " samples]";
  out.confidence = confidence;
  meta["decode"] = decode_reason;
  out.json = meta.dump();
}

inline void set_decoded_result(AsrResult& out, std::string text, float confidence,
                               size_t emitted_tokens, nlohmann::json& meta) {
  out.text = std::move(text);
  out.confidence = confidence;
  meta["emitted_tokens"] = emitted_tokens;
  meta["decode"] = out.text.empty() ? "no_text" : "ok";
  out.json = meta.dump();
}

inline std::string format_token_id_preview(std::string const& tag, std::vector<int64_t> const& ids,
                                           size_t preview_limit = 8) {
  if (ids.empty()) {
    return "";
  }
  std::string text = "[" + tag + " ids:";
  size_t preview = std::min(ids.size(), preview_limit);
  for (size_t i = 0; i < preview; ++i) {
    text += " " + std::to_string(ids[i]);
  }
  if (ids.size() > preview) {
    text += " ...";
  }
  text += "]";
  return text;
}

}  // namespace ears::asr_onnx
