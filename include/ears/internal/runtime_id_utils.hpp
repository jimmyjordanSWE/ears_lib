#pragma once

#include <string>
#include <vector>

#include "ears/internal/string_utils.hpp"

namespace ears::internal {

inline std::string normalize_runtime_id(std::string runtime) {
  runtime = to_lower_ascii(trim_copy(runtime));
  if (runtime.empty() || runtime == "auto" || runtime == "onnx") {
    return "onnx_runtime";
  }
  return runtime;
}

inline std::string normalize_provider_id(std::string provider) {
  provider = to_lower_ascii(trim_copy(provider));
  if (provider == "dml") {
    return "directml";
  }
  if (provider == "coreml") {
    return "coreml_ep";
  }
  if (provider == "openvino") {
    return "openvino_device";
  }
  if (provider == "qnn") {
    return "qnn_ep";
  }
  return provider;
}

inline std::string default_provider_for_runtime(std::string const& runtime_id) {
  std::string runtime = normalize_runtime_id(runtime_id);
  if (runtime == "tensorrt") {
    return "tensorrt";
  }
  if (runtime == "openvino") {
    return "openvino_device";
  }
  if (runtime == "coreml") {
    return "coreml_ep";
  }
  if (runtime == "qnn") {
    return "qnn_ep";
  }
  return "auto";
}

inline std::vector<std::string> runtime_default_provider_chain(std::string const& runtime_id) {
  std::string runtime = normalize_runtime_id(runtime_id);
  if (runtime == "tensorrt") {
    return {"tensorrt", "cuda", "cpu"};
  }
  if (runtime == "openvino") {
    return {"openvino_device", "cpu"};
  }
  if (runtime == "coreml") {
    return {"coreml_ep", "cpu"};
  }
  if (runtime == "qnn") {
    return {"qnn_ep", "cpu"};
  }
  return {"cuda", "directml", "coreml_ep", "openvino_device", "qnn_ep", "migraphx", "cpu"};
}

}  // namespace ears::internal
