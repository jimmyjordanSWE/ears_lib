#include "ears/factory.hpp"

#include <algorithm>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "ears/asr/dummy_asr.hpp"
#include "ears/asr/native_runtime_asr_adapter.hpp"
#include "ears/internal/env_utils.hpp"
#include "ears/internal/runtime_id_utils.hpp"
#include "ears/internal/string_utils.hpp"
#include "ears/llm/dummy_llm.hpp"
#include "ears/vad/dummy_vad.hpp"
#include "factory_internal.hpp"

namespace ears {

namespace {

#ifndef EARS_HAS_ONNX
#define EARS_HAS_ONNX 0
#endif
#ifndef EARS_HAS_TENSORRT
#define EARS_HAS_TENSORRT 0
#endif
#ifndef EARS_HAS_OPENVINO
#define EARS_HAS_OPENVINO 0
#endif
#ifndef EARS_HAS_COREML
#define EARS_HAS_COREML 0
#endif
#ifndef EARS_HAS_QNN
#define EARS_HAS_QNN 0
#endif

using factory_internal::AsrProviderRegistry;
using factory_internal::AsrRegistry;
using factory_internal::AsrRuntimeRegistry;
using factory_internal::LlmRegistry;
using factory_internal::VadRegistry;

VadRegistry& vad_registry() {
  static VadRegistry registry;
  return registry;
}

AsrRegistry& asr_registry() {
  static AsrRegistry registry;
  return registry;
}

AsrRuntimeRegistry& asr_runtime_registry() {
  static AsrRuntimeRegistry registry;
  return registry;
}

AsrProviderRegistry& asr_provider_registry() {
  static AsrProviderRegistry registry;
  return registry;
}

LlmRegistry& llm_registry() {
  static LlmRegistry registry;
  return registry;
}

std::vector<std::string>& backend_registration_trace() {
  static std::vector<std::string> trace;
  return trace;
}

std::mutex& registry_mutex() {
  static std::mutex mutex;
  return mutex;
}

bool& core_builtins_ready() {
  static bool ready = false;
  return ready;
}

std::vector<std::string> compiled_asr_runtimes() {
  std::vector<std::string> runtimes;
#if EARS_HAS_ONNX
  runtimes.emplace_back("onnx_runtime");
#endif
#if EARS_HAS_TENSORRT
  runtimes.emplace_back("tensorrt");
#endif
#if EARS_HAS_OPENVINO
  runtimes.emplace_back("openvino");
#endif
#if EARS_HAS_COREML
  runtimes.emplace_back("coreml");
#endif
#if EARS_HAS_QNN
  runtimes.emplace_back("qnn");
#endif
  return runtimes;
}

bool has_compiled_asr_runtime(std::string const& runtime_id) {
  std::string const runtime = internal::normalize_runtime_id(runtime_id);
  if (runtime == "onnx_runtime") {
    return EARS_HAS_ONNX != 0;
  }
  if (runtime == "tensorrt") {
    return EARS_HAS_TENSORRT != 0;
  }
  if (runtime == "openvino") {
    return EARS_HAS_OPENVINO != 0;
  }
  if (runtime == "coreml") {
    return EARS_HAS_COREML != 0;
  }
  if (runtime == "qnn") {
    return EARS_HAS_QNN != 0;
  }
  return false;
}

bool is_known_runtime_id(std::string const& runtime_id) {
  return runtime_id == "onnx_runtime" || runtime_id == "tensorrt" || runtime_id == "openvino" ||
         runtime_id == "coreml" || runtime_id == "qnn";
}

std::string infer_asr_family_from_model(std::string const& model_id) {
  std::string const lowered = internal::to_lower_ascii(model_id);
  if (lowered.find("hybrid") != std::string::npos ||
      lowered.find("ctc_attention") != std::string::npos) {
    return "hybrid_ctc_attention";
  }
  if (lowered.find("stream") != std::string::npos && lowered.find("seq2seq") != std::string::npos) {
    return "streaming_seq2seq";
  }
  if (lowered.find("rnnt") != std::string::npos ||
      lowered.find("transducer") != std::string::npos) {
    return "rnnt_transducer";
  }
  if (lowered.find("ctc") != std::string::npos || lowered.find("parakeet") != std::string::npos ||
      lowered.find("wav2vec") != std::string::npos) {
    return "ctc";
  }
  if (lowered.find("whisper") != std::string::npos) {
    return "whisper_seq2seq";
  }
  if (lowered.find("moonshine") != std::string::npos) {
    return "moonshine_seq2seq";
  }
  return "";
}

class StatusAsr final : public IAutomaticSpeechRecognizer {
public:
  StatusAsr(std::string code, std::string runtime_id, std::string family_id)
      : code_(std::move(code)),
        runtime_id_(std::move(runtime_id)),
        family_id_(std::move(family_id)) {}

  AsrResult recognize(float const* /*samples*/, size_t /*num_samples*/) override {
    AsrResult result;
    result.text = "[asr_error:" + code_ + "]";
    result.json = "{\"status\":\"error\",\"code\":\"" + code_ + "\",\"runtime\":\"" + runtime_id_ +
                  "\",\"family\":\"" + family_id_ + "\"}";
    result.confidence = 0.0f;
    return result;
  }

private:
  std::string code_;
  std::string runtime_id_;
  std::string family_id_;
};

enum class NativeRunnerUnavailablePolicy {
  adapter,
  model,
  dummy,
  error,
};

NativeRunnerUnavailablePolicy native_runner_unavailable_policy() {
  std::string raw = internal::to_lower_ascii(
      internal::trim_copy(internal::getenv_copy("EARS_NATIVE_RUNNER_UNAVAILABLE_POLICY")));
  if (raw.empty() || raw == "adapter" || raw == "none") {
    return NativeRunnerUnavailablePolicy::adapter;
  }
  if (raw == "error" || raw == "fail") {
    return NativeRunnerUnavailablePolicy::error;
  }
  if (raw == "dummy") {
    return NativeRunnerUnavailablePolicy::dummy;
  }
  if (raw == "model") {
    return NativeRunnerUnavailablePolicy::model;
  }
  return NativeRunnerUnavailablePolicy::adapter;
}

template <typename RegistryT>
std::unordered_set<std::string> registry_key_set_locked(RegistryT const& registry) {
  std::unordered_set<std::string> keys;
  keys.reserve(registry.size());
  for (auto const& entry : registry) {
    keys.insert(entry.first);
  }
  return keys;
}

template <typename RegistryT>
void append_new_registrations_to_trace_locked(std::string const& component,
                                              std::unordered_set<std::string> const& before,
                                              RegistryT const& after) {
  std::vector<std::string> added;
  for (auto const& entry : after) {
    if (before.find(entry.first) == before.end()) {
      added.push_back(entry.first);
    }
  }
  std::sort(added.begin(), added.end());
  for (std::string const& key : added) {
    std::string entry = "backend_register component=";
    entry += component;
    entry += " key=";
    entry += key;
    entry += " source=factory_backends.cpp";
    backend_registration_trace().push_back(std::move(entry));
  }
}

void register_builtin_factories_locked() {
  if (core_builtins_ready()) {
    return;
  }

  backend_registration_trace().clear();
  vad_registry()["dummy"] = [](Config const&) { return std::make_unique<DummyVad>(1.f); };
  backend_registration_trace().push_back(
      "backend_register component=vad key=dummy source=factory.cpp");
  asr_registry()["dummy"] = [](Config const&) { return std::make_unique<DummyAsr>(); };
  backend_registration_trace().push_back(
      "backend_register component=asr key=dummy source=factory.cpp");
  llm_registry()["dummy"] = [](Config const&) { return std::make_unique<DummyLlm>(); };
  backend_registration_trace().push_back(
      "backend_register component=llm key=dummy source=factory.cpp");

  auto const vad_before = registry_key_set_locked(vad_registry());
  auto const asr_before = registry_key_set_locked(asr_registry());
  auto const asr_runtime_before = registry_key_set_locked(asr_runtime_registry());
  auto const asr_provider_before = registry_key_set_locked(asr_provider_registry());
  auto const llm_before = registry_key_set_locked(llm_registry());

  factory_internal::register_builtin_backend_factories(vad_registry(), asr_registry(),
                                                       asr_runtime_registry(),
                                                       asr_provider_registry(), llm_registry());
  append_new_registrations_to_trace_locked("vad", vad_before, vad_registry());
  append_new_registrations_to_trace_locked("asr", asr_before, asr_registry());
  append_new_registrations_to_trace_locked("asr_runtime", asr_runtime_before,
                                           asr_runtime_registry());
  append_new_registrations_to_trace_locked("asr_provider", asr_provider_before,
                                           asr_provider_registry());
  append_new_registrations_to_trace_locked("llm", llm_before, llm_registry());

  core_builtins_ready() = true;
}

template <typename RegistryT>
std::vector<std::string> list_registry_keys_locked(RegistryT const& registry) {
  std::vector<std::string> keys;
  keys.reserve(registry.size());
  for (auto const& entry : registry) {
    keys.push_back(entry.first);
  }
  std::sort(keys.begin(), keys.end());
  return keys;
}

}  // namespace

bool register_vad_factory(std::string const& model_id, VadFactoryFn factory) {
  if (model_id.empty() || !factory) {
    return false;
  }
  std::lock_guard<std::mutex> lock(registry_mutex());
  register_builtin_factories_locked();
  vad_registry()[internal::to_lower_ascii(model_id)] = std::move(factory);
  return true;
}

bool register_asr_factory(std::string const& model_id, AsrFactoryFn factory) {
  if (model_id.empty() || !factory) {
    return false;
  }
  std::lock_guard<std::mutex> lock(registry_mutex());
  register_builtin_factories_locked();
  asr_registry()[internal::to_lower_ascii(model_id)] = std::move(factory);
  return true;
}

bool register_asr_runtime_factory(std::string const& model_family, std::string const& runtime_id,
                                  AsrFactoryFn factory) {
  if (model_family.empty() || runtime_id.empty() || !factory) {
    return false;
  }
  std::lock_guard<std::mutex> lock(registry_mutex());
  register_builtin_factories_locked();
  asr_runtime_registry()[factory_internal::asr_runtime_key(model_family, runtime_id)] =
      std::move(factory);
  return true;
}

bool register_asr_provider_factory(std::string const& model_family, std::string const& runtime_id,
                                   std::string const& provider_id, AsrFactoryFn factory) {
  if (model_family.empty() || runtime_id.empty() || provider_id.empty() || !factory) {
    return false;
  }
  std::lock_guard<std::mutex> lock(registry_mutex());
  register_builtin_factories_locked();
  asr_provider_registry()[factory_internal::asr_provider_key(model_family, runtime_id,
                                                             provider_id)] = std::move(factory);
  return true;
}

bool register_llm_factory(std::string const& model_id, LlmFactoryFn factory) {
  if (model_id.empty() || !factory) {
    return false;
  }
  std::lock_guard<std::mutex> lock(registry_mutex());
  register_builtin_factories_locked();
  llm_registry()[internal::to_lower_ascii(model_id)] = std::move(factory);
  return true;
}

void reset_factory_registry() {
  std::lock_guard<std::mutex> lock(registry_mutex());
  vad_registry().clear();
  asr_registry().clear();
  asr_runtime_registry().clear();
  asr_provider_registry().clear();
  llm_registry().clear();
  backend_registration_trace().clear();
  core_builtins_ready() = false;
}

std::vector<std::string> list_registered_vad_factories() {
  std::lock_guard<std::mutex> lock(registry_mutex());
  register_builtin_factories_locked();
  return list_registry_keys_locked(vad_registry());
}

std::vector<std::string> list_registered_asr_factories() {
  std::lock_guard<std::mutex> lock(registry_mutex());
  register_builtin_factories_locked();
  return list_registry_keys_locked(asr_registry());
}

std::vector<std::string> list_registered_asr_runtime_factories() {
  std::lock_guard<std::mutex> lock(registry_mutex());
  register_builtin_factories_locked();
  return list_registry_keys_locked(asr_runtime_registry());
}

std::vector<std::string> list_registered_asr_provider_factories() {
  std::lock_guard<std::mutex> lock(registry_mutex());
  register_builtin_factories_locked();
  return list_registry_keys_locked(asr_provider_registry());
}

std::vector<std::string> list_registered_llm_factories() {
  std::lock_guard<std::mutex> lock(registry_mutex());
  register_builtin_factories_locked();
  return list_registry_keys_locked(llm_registry());
}

std::vector<std::string> list_compiled_asr_runtimes() {
  return compiled_asr_runtimes();
}

std::vector<std::string> get_backend_registration_trace() {
  std::lock_guard<std::mutex> lock(registry_mutex());
  register_builtin_factories_locked();
  return backend_registration_trace();
}

bool has_asr_runtime_capability(std::string const& runtime_id) {
  return has_compiled_asr_runtime(runtime_id);
}

bool has_asr_runtime_family_capability(std::string const& model_family,
                                       std::string const& runtime_id) {
  if (model_family.empty() || runtime_id.empty()) {
    return false;
  }
  std::lock_guard<std::mutex> lock(registry_mutex());
  register_builtin_factories_locked();
  std::string const key = factory_internal::asr_runtime_key(model_family, runtime_id);
  return asr_runtime_registry().find(key) != asr_runtime_registry().end();
}

bool has_asr_runtime_provider_capability(std::string const& model_family,
                                         std::string const& runtime_id,
                                         std::string const& provider_id) {
  if (model_family.empty() || runtime_id.empty() || provider_id.empty()) {
    return false;
  }
  std::lock_guard<std::mutex> lock(registry_mutex());
  register_builtin_factories_locked();
  std::string const key = factory_internal::asr_provider_key(model_family, runtime_id, provider_id);
  return asr_provider_registry().find(key) != asr_provider_registry().end();
}

std::unique_ptr<IVoiceActivityDetector> create_vad(Config const& config) {
  VadFactoryFn selected;
  VadFactoryFn fallback;
  {
    std::lock_guard<std::mutex> lock(registry_mutex());
    register_builtin_factories_locked();
    auto const selected_it = vad_registry().find(internal::to_lower_ascii(config.vad.model));
    if (selected_it != vad_registry().end()) {
      selected = selected_it->second;
    }
    auto const fallback_it = vad_registry().find("dummy");
    if (fallback_it != vad_registry().end()) {
      fallback = fallback_it->second;
    }
  }

  try {
    if (selected) {
      auto instance = selected(config);
      if (instance) {
        return instance;
      }
    }
  } catch (...) {
    selected = nullptr;
  }

  try {
    if (fallback) {
      auto instance = fallback(config);
      if (instance) {
        return instance;
      }
    }
  } catch (...) {
    fallback = nullptr;
  }
  return std::make_unique<DummyVad>(1.f);
}

std::unique_ptr<IAutomaticSpeechRecognizer> create_asr(Config const& config) {
  AsrFactoryFn provider_selected;
  AsrFactoryFn runtime_selected;
  AsrFactoryFn selected;
  AsrFactoryFn fallback;
  std::string effective_family;
  std::string effective_runtime;
  bool runtime_explicit = false;
  {
    std::lock_guard<std::mutex> lock(registry_mutex());
    register_builtin_factories_locked();

    runtime_explicit = !internal::trim_copy(config.asr.runtime).empty() &&
                       internal::to_lower_ascii(internal::trim_copy(config.asr.runtime)) != "auto";
    effective_family = config.asr.family.empty() ? infer_asr_family_from_model(config.asr.model)
                                                 : internal::to_lower_ascii(config.asr.family);
    effective_runtime = factory_internal::resolve_asr_runtime(config);
    std::string const effective_provider =
        factory_internal::resolve_asr_provider(config, effective_runtime);

    if (!effective_family.empty()) {
      auto const provider_it = asr_provider_registry().find(factory_internal::asr_provider_key(
          effective_family, effective_runtime, effective_provider));
      if (provider_it != asr_provider_registry().end()) {
        provider_selected = provider_it->second;
      }

      auto const runtime_it = asr_runtime_registry().find(
          factory_internal::asr_runtime_key(effective_family, effective_runtime));
      if (runtime_it != asr_runtime_registry().end()) {
        runtime_selected = runtime_it->second;
      }
    }

    auto const selected_it = asr_registry().find(internal::to_lower_ascii(config.asr.model));
    if (selected_it != asr_registry().end()) {
      selected = selected_it->second;
    }
    auto const fallback_it = asr_registry().find("dummy");
    if (fallback_it != asr_registry().end()) {
      fallback = fallback_it->second;
    }
  }

  if (runtime_explicit && is_known_runtime_id(effective_runtime) &&
      !has_compiled_asr_runtime(effective_runtime)) {
    return std::make_unique<StatusAsr>("runtime_unavailable", effective_runtime, effective_family);
  }

  if (runtime_explicit && !effective_family.empty() && is_known_runtime_id(effective_runtime) &&
      has_compiled_asr_runtime(effective_runtime) &&
      !has_asr_runtime_family_capability(effective_family, effective_runtime)) {
    return std::make_unique<StatusAsr>("unsupported_family_runtime", effective_runtime,
                                       effective_family);
  }

  if (runtime_explicit && is_native_runtime_id(effective_runtime)) {
    NativeRuntimeRunnerProbe const probe = probe_native_runtime_runner(effective_runtime);
    if (!probe.healthy) {
      NativeRunnerUnavailablePolicy const policy = native_runner_unavailable_policy();
      if (policy == NativeRunnerUnavailablePolicy::adapter) {
        // Keep adapter path active; runtime bridge will emit runner diagnostics at recognize-time.
      } else if (policy == NativeRunnerUnavailablePolicy::error) {
        return std::make_unique<StatusAsr>("runner_unavailable", effective_runtime,
                                           effective_family);
      } else if (policy == NativeRunnerUnavailablePolicy::model) {
        provider_selected = nullptr;
        runtime_selected = nullptr;
      } else if (policy == NativeRunnerUnavailablePolicy::dummy) {
        provider_selected = nullptr;
        runtime_selected = nullptr;
        selected = nullptr;
      }
    }
  }

  try {
    if (provider_selected) {
      auto instance = provider_selected(config);
      if (instance) {
        return instance;
      }
    }
  } catch (...) {
    provider_selected = nullptr;
  }

  try {
    if (runtime_selected) {
      auto instance = runtime_selected(config);
      if (instance) {
        return instance;
      }
    }
  } catch (...) {
    runtime_selected = nullptr;
  }

  try {
    if (selected) {
      auto instance = selected(config);
      if (instance) {
        return instance;
      }
    }
  } catch (...) {
    selected = nullptr;
  }

  try {
    if (fallback) {
      auto instance = fallback(config);
      if (instance) {
        return instance;
      }
    }
  } catch (...) {
    fallback = nullptr;
  }

  return std::make_unique<DummyAsr>();
}

std::unique_ptr<ISemanticRectifier> create_llm(Config const& config) {
  if (!config.llm.enabled) {
    return std::make_unique<DummyLlm>();
  }

  LlmFactoryFn selected;
  LlmFactoryFn fallback;
  {
    std::lock_guard<std::mutex> lock(registry_mutex());
    register_builtin_factories_locked();
    auto const selected_it = llm_registry().find(internal::to_lower_ascii(config.llm.model));
    if (selected_it != llm_registry().end()) {
      selected = selected_it->second;
    }
    auto const fallback_it = llm_registry().find("dummy");
    if (fallback_it != llm_registry().end()) {
      fallback = fallback_it->second;
    }
  }

  try {
    if (selected) {
      auto instance = selected(config);
      if (instance) {
        return instance;
      }
    }
  } catch (...) {
    selected = nullptr;
  }

  try {
    if (fallback) {
      auto instance = fallback(config);
      if (instance) {
        return instance;
      }
    }
  } catch (...) {
    fallback = nullptr;
  }

  return std::make_unique<DummyLlm>();
}

}  // namespace ears
