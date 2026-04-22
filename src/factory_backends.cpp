#include "ears/asr/dummy_asr.hpp"
#include "factory_internal.hpp"

#if EARS_HAS_TENSORRT
#include "ears/asr/tensorrt_asr_adapter.hpp"
#endif
#if EARS_HAS_OPENVINO
#include "ears/asr/openvino_asr_adapter.hpp"
#endif
#if EARS_HAS_COREML
#include "ears/asr/coreml_asr_adapter.hpp"
#endif
#if EARS_HAS_QNN
#include "ears/asr/qnn_asr_adapter.hpp"
#endif

#if EARS_HAS_ONNX
#include "ears/asr/ctc_asr_adapter.hpp"
#include "ears/asr/hybrid_ctc_attention_adapter.hpp"
#include "ears/asr/moonshine_asr.hpp"
#include "ears/asr/rnnt_asr_adapter.hpp"
#include "ears/asr/streaming_seq2seq_adapter.hpp"
#include "ears/asr/whisper_seq2seq_adapter.hpp"
#include "ears/llm/smollm_rectifier.hpp"
#include "ears/vad/silero_vad.hpp"
#endif

#include <string>
#include <utility>
#include <vector>

namespace ears::factory_internal {

namespace {

std::vector<std::string> const kKnownAsrFamilies = {"moonshine_seq2seq",    "whisper_seq2seq",
                                                    "streaming_seq2seq",    "ctc",
                                                    "hybrid_ctc_attention", "rnnt",
                                                    "rnnt_transducer"};

AsrFactoryFn make_runtime_stub_factory(std::string family_id, std::string runtime_id) {
  std::string const family = internal::to_lower_ascii(std::move(family_id));
  std::string const runtime = internal::normalize_runtime_id(std::move(runtime_id));

  return [family, runtime](Config const& config) -> std::unique_ptr<IAutomaticSpeechRecognizer> {
#if EARS_HAS_TENSORRT
    if (runtime == "tensorrt") {
      return std::make_unique<TensorRtAsrAdapter>(family, config.asr.path, 16000);
    }
#endif
#if EARS_HAS_OPENVINO
    if (runtime == "openvino") {
      return std::make_unique<OpenVinoAsrAdapter>(family, config.asr.path, 16000);
    }
#endif
#if EARS_HAS_COREML
    if (runtime == "coreml") {
      return std::make_unique<CoreMlAsrAdapter>(family, config.asr.path, 16000);
    }
#endif
#if EARS_HAS_QNN
    if (runtime == "qnn") {
      return std::make_unique<QnnAsrAdapter>(family, config.asr.path, 16000);
    }
#endif
    (void)family;
    (void)config;
    return std::make_unique<DummyAsr>();
  };
}

void register_native_runtime_stubs(AsrRuntimeRegistry& asr_runtime) {
  std::vector<std::string> runtimes;
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
  for (std::string const& family : kKnownAsrFamilies) {
    for (std::string const& runtime : runtimes) {
      asr_runtime[asr_runtime_key(family, runtime)] = make_runtime_stub_factory(family, runtime);
    }
  }
}

#if EARS_HAS_ONNX

AsrFactoryFn make_onnx_family_factory(std::string family_id, std::string forced_provider = "") {
  std::string const family = internal::to_lower_ascii(std::move(family_id));
  std::string const provider_override = internal::normalize_provider_id(std::move(forced_provider));

  return [family,
          provider_override](Config const& config) -> std::unique_ptr<IAutomaticSpeechRecognizer> {
    std::string provider = provider_override;
    if (provider.empty()) {
      provider = resolve_asr_provider(config, "onnx_runtime");
    }

    if (family == "whisper_seq2seq") {
      return std::make_unique<WhisperSeq2SeqAdapter>(config.asr.path, 16000, provider);
    }
    if (family == "streaming_seq2seq") {
      return std::make_unique<StreamingSeq2SeqAdapter>(config.asr.path, 16000, provider);
    }
    if (family == "ctc") {
      return std::make_unique<CtcAsrAdapter>(config.asr.path, 16000, provider);
    }
    if (family == "hybrid_ctc_attention") {
      return std::make_unique<HybridCtcAttentionAdapter>(config.asr.path, 16000, provider);
    }
    if (family == "rnnt" || family == "rnnt_transducer") {
      return std::make_unique<RnntAsrAdapter>(config.asr.path, 16000, provider);
    }
    if (family == "moonshine_seq2seq") {
      return std::make_unique<MoonshineAsr>(config.asr.path, 16000);
    }
    return std::make_unique<DummyAsr>();
  };
}

void register_onnx_backends(VadRegistry& vad, AsrRegistry& asr, AsrRuntimeRegistry& asr_runtime,
                            AsrProviderRegistry& asr_provider, LlmRegistry& llm) {
  vad["silero"] = [](Config const& config) -> std::unique_ptr<IVoiceActivityDetector> {
    return std::make_unique<SileroVad>(config.vad.path, 16000);
  };

  asr["moonshine"] = [](Config const& config) -> std::unique_ptr<IAutomaticSpeechRecognizer> {
    return std::make_unique<MoonshineAsr>(config.asr.path, 16000);
  };
  asr["whisper"] = [](Config const& config) -> std::unique_ptr<IAutomaticSpeechRecognizer> {
    return std::make_unique<WhisperSeq2SeqAdapter>(
        config.asr.path, 16000, internal::normalize_provider_id(config.provider));
  };
  asr["whisper_seq2seq"] = asr["whisper"];
  asr["streaming_seq2seq"] =
      [](Config const& config) -> std::unique_ptr<IAutomaticSpeechRecognizer> {
    return std::make_unique<StreamingSeq2SeqAdapter>(
        config.asr.path, 16000, internal::normalize_provider_id(config.provider));
  };
  asr["ctc"] = [](Config const& config) -> std::unique_ptr<IAutomaticSpeechRecognizer> {
    return std::make_unique<CtcAsrAdapter>(config.asr.path, 16000,
                                           internal::normalize_provider_id(config.provider));
  };
  asr["hybrid_ctc_attention"] =
      [](Config const& config) -> std::unique_ptr<IAutomaticSpeechRecognizer> {
    return std::make_unique<HybridCtcAttentionAdapter>(
        config.asr.path, 16000, internal::normalize_provider_id(config.provider));
  };
  asr["rnnt"] = [](Config const& config) -> std::unique_ptr<IAutomaticSpeechRecognizer> {
    return std::make_unique<RnntAsrAdapter>(config.asr.path, 16000,
                                            internal::normalize_provider_id(config.provider));
  };
  asr["rnnt_transducer"] = asr["rnnt"];

  std::vector<std::string> const onnx_runtimes = {"onnx_runtime", "onnx"};
  for (std::string const& family : kKnownAsrFamilies) {
    for (std::string const& runtime : onnx_runtimes) {
      (void)runtime;
      asr_runtime[asr_runtime_key(family, "onnx_runtime")] = make_onnx_family_factory(family);
    }
  }

  std::vector<std::string> const onnx_providers = {
      "cuda", "directml", "coreml_ep", "openvino_device", "qnn_ep", "tensorrt"};

  for (std::string const& family : kKnownAsrFamilies) {
    for (std::string const& provider : onnx_providers) {
      asr_provider[asr_provider_key(family, "onnx_runtime", provider)] =
          make_onnx_family_factory(family, provider);
    }
  }

  llm["smollm2"] = [](Config const& config) -> std::unique_ptr<ISemanticRectifier> {
    return std::make_unique<SmolLMRectifier>(config.llm.path);
  };
  llm["smollm"] = llm["smollm2"];
}

#endif  // EARS_HAS_ONNX

}  // namespace

void register_builtin_backend_factories(VadRegistry& vad, AsrRegistry& asr,
                                        AsrRuntimeRegistry& asr_runtime,
                                        AsrProviderRegistry& asr_provider, LlmRegistry& llm) {
  register_native_runtime_stubs(asr_runtime);
#if EARS_HAS_ONNX
  register_onnx_backends(vad, asr, asr_runtime, asr_provider, llm);
#else
  (void)vad;
  (void)asr;
  (void)asr_provider;
  (void)llm;
#endif
}

}  // namespace ears::factory_internal
