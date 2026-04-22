#pragma once

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <limits>
#include <nlohmann/json.hpp>
#include <onnxruntime_cxx_api.h>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "ears/internal/runtime_id_utils.hpp"
#include "ears/internal/string_utils.hpp"
#include "ears/internal/unicode_utils.hpp"

namespace ears::asr_onnx {

inline std::string to_lower_ascii(std::string value) {
  return internal::to_lower_ascii(std::move(value));
}

inline std::string trim_copy(std::string const& input) {
  return internal::trim_copy(input);
}

inline void replace_all(std::string& text, std::string const& from, std::string const& to) {
  if (from.empty()) {
    return;
  }
  size_t pos = 0;
  while ((pos = text.find(from, pos)) != std::string::npos) {
    text.replace(pos, from.size(), to);
    pos += to.size();
  }
}

inline std::wstring to_wide(std::string const& s) {
  return internal::to_wide_utf8(s);
}

inline bool fs_path_exists(std::filesystem::path const& path) {
  std::error_code ec;
  return std::filesystem::exists(path, ec);
}

inline bool fs_is_regular_file(std::filesystem::path const& path) {
  std::error_code ec;
  return std::filesystem::is_regular_file(path, ec);
}

inline bool fs_is_directory(std::filesystem::path const& path) {
  std::error_code ec;
  return std::filesystem::is_directory(path, ec);
}

inline std::string normalize_provider_id(std::string provider) {
  return internal::normalize_provider_id(std::move(provider));
}

inline std::string normalize_runtime_id(std::string runtime) {
  return internal::normalize_runtime_id(std::move(runtime));
}

inline std::vector<std::string> runtime_default_provider_chain(std::string const& runtime) {
  return internal::runtime_default_provider_chain(runtime);
}

inline bool provider_available(std::string const& onnx_provider_name) {
  std::vector<std::string> available = Ort::GetAvailableProviders();
  return std::find(available.begin(), available.end(), onnx_provider_name) != available.end();
}

inline std::string append_onnx_provider_or_cpu(Ort::SessionOptions& opts,
                                               std::string const& runtime_id,
                                               std::string const& requested_provider) {
  auto try_provider = [&](std::string normalized_provider) -> bool {
    if (normalized_provider.empty() || normalized_provider == "cpu") {
      return false;
    }
    if (normalized_provider == "cuda") {
      if (!provider_available("CUDAExecutionProvider")) {
        return false;
      }
      OrtCUDAProviderOptions cuda_opts{};
      cuda_opts.device_id = 0;
      opts.AppendExecutionProvider_CUDA(cuda_opts);
      return true;
    }
    if (normalized_provider == "directml") {
      if (!provider_available("DmlExecutionProvider")) {
        return false;
      }
      std::unordered_map<std::string, std::string> dml_opts;
      opts.AppendExecutionProvider("DML", dml_opts);
      return true;
    }
    if (normalized_provider == "coreml_ep") {
      if (!provider_available("CoreMLExecutionProvider")) {
        return false;
      }
      std::unordered_map<std::string, std::string> coreml_opts;
      opts.AppendExecutionProvider("CoreML", coreml_opts);
      return true;
    }
    if (normalized_provider == "tensorrt") {
      if (!provider_available("TensorrtExecutionProvider")) {
        return false;
      }
      std::unordered_map<std::string, std::string> trt_opts;
      opts.AppendExecutionProvider("NvTensorRtRtx", trt_opts);
      return true;
    }
    if (normalized_provider == "migraphx") {
      if (!provider_available("MIGraphXExecutionProvider")) {
        return false;
      }
      std::unordered_map<std::string, std::string> mgx_opts;
      opts.AppendExecutionProvider("MIGraphX", mgx_opts);
      return true;
    }
    if (normalized_provider == "openvino_device") {
      if (!provider_available("OpenVINOExecutionProvider")) {
        return false;
      }
      std::unordered_map<std::string, std::string> ov_opts;
      opts.AppendExecutionProvider("OpenVINO", ov_opts);
      return true;
    }
    if (normalized_provider == "qnn_ep") {
      if (!provider_available("QNNExecutionProvider")) {
        return false;
      }
      std::unordered_map<std::string, std::string> qnn_opts;
      opts.AppendExecutionProvider("QNN", qnn_opts);
      return true;
    }
    return false;
  };

  std::string normalized_requested = normalize_provider_id(requested_provider);
  if (!normalized_requested.empty() && normalized_requested != "auto") {
    if (try_provider(normalized_requested)) {
      return normalized_requested;
    }
    throw std::runtime_error("Requested ONNX execution provider unavailable: " +
                             normalized_requested);
  }

  for (std::string const& candidate : runtime_default_provider_chain(runtime_id)) {
    if (try_provider(candidate)) {
      return normalize_provider_id(candidate);
    }
  }
  throw std::runtime_error("No supported accelerated ONNX execution provider available");
}

inline std::string resolve_model_file(std::string const& path,
                                      std::vector<std::string> const& preferred_names) {
  namespace fs = std::filesystem;
  fs::path candidate(path);
  if (fs_is_regular_file(candidate)) {
    return candidate.string();
  }
  if (!fs_is_directory(candidate)) {
    return path;
  }

  for (auto const& preferred : preferred_names) {
    fs::path preferred_path = candidate / preferred;
    if (fs_is_regular_file(preferred_path)) {
      return preferred_path.string();
    }
  }

  fs::path nested_onnx = candidate / "onnx" / "model.onnx";
  if (fs_is_regular_file(nested_onnx)) {
    return nested_onnx.string();
  }

  std::error_code ec;
  for (auto const& entry : fs::directory_iterator(candidate, ec)) {
    if (ec) {
      break;
    }
    if (!entry.is_regular_file()) {
      continue;
    }
    if (to_lower_ascii(entry.path().extension().string()) == ".onnx") {
      return entry.path().string();
    }
  }

  return path;
}

inline std::string model_base_dir(std::string const& original_path,
                                  std::string const& resolved_model_file) {
  namespace fs = std::filesystem;
  fs::path original(original_path);
  if (fs_is_directory(original)) {
    return original.string();
  }
  fs::path resolved(resolved_model_file);
  if (resolved.has_parent_path()) {
    return resolved.parent_path().string();
  }
  return ".";
}

inline bool is_wordpiece_special(std::string const& token) {
  if (token.size() >= 2 && token.front() == '<' && token.back() == '>') {
    return true;
  }
  if (token.size() >= 2 && token.front() == '[' && token.back() == ']') {
    return true;
  }
  return false;
}

inline std::string normalize_token_piece(std::string token) {
  token = trim_copy(token);
  if (token.empty()) {
    return "";
  }

  std::string const lowered = to_lower_ascii(token);
  if (lowered == "<blank>" || lowered == "<ctc_blank>" || lowered == "<pad>" ||
      lowered == "[blank]" || lowered == "[pad]" || lowered == "<blk>" || lowered == "<eps>" ||
      lowered == "<s>" || lowered == "</s>" || lowered == "<unk>" || lowered == "<noise>") {
    return "";
  }

  if (token == "|" || lowered == "<space>") {
    return " ";
  }

  if (token.rfind("##", 0) == 0 && token.size() > 2) {
    token = token.substr(2);
  }
  if (token.size() > 2 && token.rfind("@@") == token.size() - 2) {
    token = token.substr(0, token.size() - 2);
  }
  if (token.size() > 4 && token.rfind("</w>") == token.size() - 4) {
    token = token.substr(0, token.size() - 4);
    token.push_back(' ');
  }

  replace_all(token, "\xE2\x96\x81", " ");  // sentencepiece ▁
  replace_all(token, "\xC4\xA0", " ");      // GPT-2 BPE Ġ

  if (is_wordpiece_special(token)) {
    return "";
  }

  return token;
}

inline std::string normalize_whitespace(std::string text) {
  std::string out;
  out.reserve(text.size());
  bool last_was_space = true;
  for (char c : text) {
    bool is_space = std::isspace(static_cast<unsigned char>(c)) != 0;
    if (is_space) {
      if (!last_was_space) {
        out.push_back(' ');
      }
      last_was_space = true;
    } else {
      out.push_back(c);
      last_was_space = false;
    }
  }
  return trim_copy(out);
}

inline bool parse_tokens_from_json(std::filesystem::path const& path,
                                   std::vector<std::string>& out_tokens) {
  std::ifstream in(path);
  if (!in) {
    return false;
  }

  nlohmann::json parsed;
  try {
    in >> parsed;
  } catch (...) {
    return false;
  }

  auto load_from_id_map = [&](nlohmann::json const& object) {
    if (!object.is_object()) {
      return false;
    }
    size_t max_id = 0;
    bool any = false;
    for (auto const& item : object.items()) {
      if (!item.value().is_number_integer()) {
        return false;
      }
      int id = item.value().get<int>();
      if (id < 0) {
        continue;
      }
      max_id = std::max(max_id, static_cast<size_t>(id));
      any = true;
    }
    if (!any) {
      return false;
    }
    std::vector<std::string> tokens(max_id + 1);
    for (auto const& item : object.items()) {
      if (!item.value().is_number_integer()) {
        continue;
      }
      int id = item.value().get<int>();
      if (id < 0) {
        continue;
      }
      tokens[static_cast<size_t>(id)] = item.key();
    }
    out_tokens = std::move(tokens);
    return true;
  };

  auto load_from_label_map = [&](nlohmann::json const& object) {
    if (!object.is_object()) {
      return false;
    }
    size_t max_id = 0;
    bool any = false;
    for (auto const& item : object.items()) {
      char* end_ptr = nullptr;
      long id = std::strtol(item.key().c_str(), &end_ptr, 10);
      if (end_ptr == nullptr || *end_ptr != '\0') {
        continue;
      }
      if (!item.value().is_string() || id < 0) {
        continue;
      }
      max_id = std::max(max_id, static_cast<size_t>(id));
      any = true;
    }
    if (!any) {
      return false;
    }
    std::vector<std::string> tokens(max_id + 1);
    for (auto const& item : object.items()) {
      char* end_ptr = nullptr;
      long id = std::strtol(item.key().c_str(), &end_ptr, 10);
      if (end_ptr == nullptr || *end_ptr != '\0' || id < 0 || !item.value().is_string()) {
        continue;
      }
      tokens[static_cast<size_t>(id)] = item.value().get<std::string>();
    }
    out_tokens = std::move(tokens);
    return true;
  };

  if (parsed.is_array()) {
    bool all_strings = true;
    std::vector<std::string> tokens;
    tokens.reserve(parsed.size());
    for (auto const& v : parsed) {
      if (!v.is_string()) {
        all_strings = false;
        break;
      }
      tokens.push_back(v.get<std::string>());
    }
    if (all_strings && !tokens.empty()) {
      out_tokens = std::move(tokens);
      return true;
    }
  }

  if (parsed.is_object()) {
    if (load_from_id_map(parsed)) {
      return true;
    }
    if (parsed.contains("id2label") && load_from_label_map(parsed["id2label"])) {
      return true;
    }
    if (parsed.contains("vocab") && load_from_id_map(parsed["vocab"])) {
      return true;
    }
    if (parsed.contains("model") && parsed["model"].is_object() &&
        parsed["model"].contains("vocab") && load_from_id_map(parsed["model"]["vocab"])) {
      return true;
    }
  }

  return false;
}

inline bool parse_tokens_from_text(std::filesystem::path const& path,
                                   std::vector<std::string>& out_tokens) {
  std::ifstream in(path);
  if (!in) {
    return false;
  }

  std::vector<std::pair<int, std::string>> indexed_tokens;
  std::vector<std::string> tokens;
  std::string line;
  bool saw_indexed = false;
  while (std::getline(in, line)) {
    line = trim_copy(line);
    if (line.empty()) {
      continue;
    }

    // Accept either "<index> <token>" or "<token> <index>" formats.
    size_t first_space = line.find_first_of(" \t");
    if (first_space != std::string::npos) {
      std::string maybe_idx = line.substr(0, first_space);
      std::string maybe_token = trim_copy(line.substr(first_space + 1));
      char* end_ptr = nullptr;
      long idx = std::strtol(maybe_idx.c_str(), &end_ptr, 10);
      if (end_ptr != nullptr && *end_ptr == '\0' && idx >= 0 && !maybe_token.empty()) {
        indexed_tokens.emplace_back(static_cast<int>(idx), maybe_token);
        saw_indexed = true;
        continue;
      }
    }

    size_t last_space = line.find_last_of(" \t");
    if (last_space != std::string::npos) {
      std::string maybe_token = trim_copy(line.substr(0, last_space));
      std::string maybe_idx = trim_copy(line.substr(last_space + 1));
      char* end_ptr = nullptr;
      long idx = std::strtol(maybe_idx.c_str(), &end_ptr, 10);
      if (end_ptr != nullptr && *end_ptr == '\0' && idx >= 0 && !maybe_token.empty()) {
        indexed_tokens.emplace_back(static_cast<int>(idx), maybe_token);
        saw_indexed = true;
        continue;
      }
    }

    tokens.push_back(line);
  }

  if (saw_indexed) {
    int max_idx = -1;
    for (auto const& item : indexed_tokens) {
      max_idx = std::max(max_idx, item.first);
    }
    if (max_idx >= 0) {
      std::vector<std::string> out(static_cast<size_t>(max_idx + 1));
      for (auto const& item : indexed_tokens) {
        out[static_cast<size_t>(item.first)] = item.second;
      }
      out_tokens = std::move(out);
      return true;
    }
  }

  if (!tokens.empty()) {
    out_tokens = std::move(tokens);
    return true;
  }
  return false;
}

struct TokenTable {
  std::vector<std::string> tokens;
  int blank_id = 0;
  std::string source_path;
};

inline int detect_blank_id(std::vector<std::string> const& tokens) {
  for (size_t i = 0; i < tokens.size(); ++i) {
    std::string lowered = to_lower_ascii(trim_copy(tokens[i]));
    if (lowered == "<blank>" || lowered == "<ctc_blank>" || lowered == "[blank]" ||
        lowered == "<blk>" || lowered == "<eps>") {
      return static_cast<int>(i);
    }
  }
  for (size_t i = 0; i < tokens.size(); ++i) {
    std::string lowered = to_lower_ascii(trim_copy(tokens[i]));
    if (lowered == "<pad>" || lowered == "[pad]") {
      return static_cast<int>(i);
    }
  }
  return 0;
}

inline TokenTable load_token_table(std::string const& base_dir) {
  namespace fs = std::filesystem;
  std::vector<std::string> const token_candidates = {"tokens.txt", "vocab.txt",  "vocabulary.txt",
                                                     "labels.txt", "vocab.json", "tokenizer.json",
                                                     "config.json"};

  TokenTable table;
  fs::path base(base_dir);
  for (auto const& filename : token_candidates) {
    fs::path candidate = base / filename;
    if (!fs_is_regular_file(candidate)) {
      continue;
    }
    std::vector<std::string> parsed_tokens;
    bool loaded = false;
    std::string ext = to_lower_ascii(candidate.extension().string());
    if (ext == ".json") {
      loaded = parse_tokens_from_json(candidate, parsed_tokens);
    } else {
      loaded = parse_tokens_from_text(candidate, parsed_tokens);
    }
    if (loaded && !parsed_tokens.empty()) {
      table.tokens = std::move(parsed_tokens);
      table.blank_id = detect_blank_id(table.tokens);
      table.source_path = candidate.string();
      return table;
    }
  }
  return table;
}

inline float softmax_probability(float const* row, int64_t width, int64_t selected) {
  if (row == nullptr || width <= 0 || selected < 0 || selected >= width) {
    return 0.0f;
  }
  float max_logit = -std::numeric_limits<float>::infinity();
  for (int64_t i = 0; i < width; ++i) {
    max_logit = std::max(max_logit, row[i]);
  }
  double denom = 0.0;
  for (int64_t i = 0; i < width; ++i) {
    denom += std::exp(static_cast<double>(row[i] - max_logit));
  }
  if (denom <= 0.0) {
    return 0.0f;
  }
  return static_cast<float>(std::exp(static_cast<double>(row[selected] - max_logit)) / denom);
}

inline std::string token_for_id(std::vector<std::string> const& tokens, int64_t id) {
  if (id < 0 || id >= static_cast<int64_t>(tokens.size())) {
    if (id >= 32 && id <= 126) {
      return std::string(1, static_cast<char>(id));
    }
    return "";
  }
  return normalize_token_piece(tokens[static_cast<size_t>(id)]);
}

inline std::string decode_ids_to_text(std::vector<int64_t> const& ids,
                                      std::vector<std::string> const& tokens, int blank_id,
                                      bool collapse_repeats, bool reset_repeat_on_blank) {
  std::string text;
  int64_t prev = -1;
  for (int64_t id : ids) {
    if (id == blank_id) {
      if (reset_repeat_on_blank) {
        prev = -1;
      }
      continue;
    }
    if (collapse_repeats && id == prev) {
      continue;
    }
    prev = id;
    text += token_for_id(tokens, id);
  }
  return normalize_whitespace(text);
}

inline bool has_any_substring(std::string const& text, std::initializer_list<char const*> needles) {
  std::string lowered = to_lower_ascii(text);
  for (auto const* needle : needles) {
    if (lowered.find(needle) != std::string::npos) {
      return true;
    }
  }
  return false;
}

inline std::vector<int64_t> shape_with_dynamic_as_one(std::vector<int64_t> shape) {
  for (int64_t& d : shape) {
    if (d <= 0) {
      d = 1;
    }
  }
  return shape;
}

inline int64_t element_count(std::vector<int64_t> const& shape) {
  if (shape.empty()) {
    return 0;
  }
  int64_t total = 1;
  for (int64_t d : shape) {
    if (d <= 0) {
      return 0;
    }
    if (total > std::numeric_limits<int64_t>::max() / d) {
      return 0;
    }
    total *= d;
  }
  return total;
}

inline std::string extract_first_string_tensor(Ort::Value const& value) {
  if (!value.IsTensor()) {
    return "";
  }
  auto info = value.GetTensorTypeAndShapeInfo();
  if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING) {
    return "";
  }
  size_t count = info.GetElementCount();
  if (count == 0) {
    return "";
  }
  size_t total_length = value.GetStringTensorDataLength();
  std::string buffer(total_length, '\0');
  std::vector<size_t> offsets(count, 0);
  value.GetStringTensorContent(buffer.data(), total_length, offsets.data(), offsets.size());
  size_t begin = offsets[0];
  size_t end = (count > 1) ? offsets[1] : total_length;
  if (begin >= end || end > total_length) {
    return "";
  }
  return buffer.substr(begin, end - begin);
}

inline std::vector<int64_t> extract_int_tensor_ids(Ort::Value const& value) {
  std::vector<int64_t> ids;
  if (!value.IsTensor()) {
    return ids;
  }
  auto info = value.GetTensorTypeAndShapeInfo();
  auto shape = shape_with_dynamic_as_one(info.GetShape());
  int64_t count = element_count(shape);
  if (count <= 0) {
    return ids;
  }

  auto type = info.GetElementType();
  if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
    int64_t const* data = value.GetTensorData<int64_t>();
    ids.assign(data, data + count);
    return ids;
  }
  if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
    int32_t const* data = value.GetTensorData<int32_t>();
    ids.reserve(static_cast<size_t>(count));
    for (int64_t i = 0; i < count; ++i) {
      ids.push_back(static_cast<int64_t>(data[i]));
    }
    return ids;
  }
  return ids;
}

}  // namespace ears::asr_onnx
