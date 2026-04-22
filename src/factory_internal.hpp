#pragma once

#include <string>
#include <unordered_map>

#include "ears/config.hpp"
#include "ears/factory.hpp"
#include "ears/internal/runtime_id_utils.hpp"
#include "ears/internal/string_utils.hpp"

namespace ears::factory_internal {

using VadRegistry = std::unordered_map<std::string, VadFactoryFn>;
using AsrRegistry = std::unordered_map<std::string, AsrFactoryFn>;
using AsrRuntimeRegistry = std::unordered_map<std::string, AsrFactoryFn>;
using AsrProviderRegistry = std::unordered_map<std::string, AsrFactoryFn>;
using LlmRegistry = std::unordered_map<std::string, LlmFactoryFn>;

inline std::string asr_runtime_key(std::string const& model_family, std::string const& runtime_id) {
  return internal::to_lower_ascii(model_family) + "::" + internal::normalize_runtime_id(runtime_id);
}

inline std::string asr_provider_key(std::string const& model_family, std::string const& runtime_id,
                                    std::string const& provider_id) {
  return asr_runtime_key(model_family, runtime_id) +
         "::" + internal::normalize_provider_id(provider_id);
}

inline std::string resolve_asr_runtime(Config const& config) {
  return internal::normalize_runtime_id(config.asr.runtime);
}

inline std::string resolve_asr_provider(Config const& config, std::string const& runtime_id) {
  std::string provider = internal::normalize_provider_id(config.provider);
  if (provider.empty() || provider == "auto") {
    return internal::default_provider_for_runtime(runtime_id);
  }
  if (provider == "cpu" && internal::normalize_runtime_id(runtime_id) != "onnx_runtime") {
    return internal::default_provider_for_runtime(runtime_id);
  }
  return provider;
}

void register_builtin_backend_factories(VadRegistry& vad, AsrRegistry& asr,
                                        AsrRuntimeRegistry& asr_runtime,
                                        AsrProviderRegistry& asr_provider, LlmRegistry& llm);

}  // namespace ears::factory_internal
