#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "ears/asr.hpp"
#include "ears/config.hpp"
#include "ears/llm.hpp"
#include "ears/vad.hpp"

namespace ears {

using VadFactoryFn = std::function<std::unique_ptr<IVoiceActivityDetector>(Config const&)>;
using AsrFactoryFn = std::function<std::unique_ptr<IAutomaticSpeechRecognizer>(Config const&)>;
using LlmFactoryFn = std::function<std::unique_ptr<ISemanticRectifier>(Config const&)>;

bool register_vad_factory(std::string const& model_id, VadFactoryFn factory);
bool register_asr_factory(std::string const& model_id, AsrFactoryFn factory);
bool register_asr_runtime_factory(std::string const& model_family, std::string const& runtime_id,
                                  AsrFactoryFn factory);
bool register_asr_provider_factory(std::string const& model_family, std::string const& runtime_id,
                                   std::string const& provider_id, AsrFactoryFn factory);
bool register_llm_factory(std::string const& model_id, LlmFactoryFn factory);

void reset_factory_registry();

std::vector<std::string> list_registered_vad_factories();
std::vector<std::string> list_registered_asr_factories();
std::vector<std::string> list_registered_asr_runtime_factories();
std::vector<std::string> list_registered_asr_provider_factories();
std::vector<std::string> list_registered_llm_factories();
std::vector<std::string> list_compiled_asr_runtimes();
std::vector<std::string> get_backend_registration_trace();

bool has_asr_runtime_capability(std::string const& runtime_id);
bool has_asr_runtime_family_capability(std::string const& model_family,
                                       std::string const& runtime_id);
bool has_asr_runtime_provider_capability(std::string const& model_family,
                                         std::string const& runtime_id,
                                         std::string const& provider_id);

std::unique_ptr<IVoiceActivityDetector> create_vad(Config const& config);
std::unique_ptr<IAutomaticSpeechRecognizer> create_asr(Config const& config);
std::unique_ptr<ISemanticRectifier> create_llm(Config const& config);

}  // namespace ears
