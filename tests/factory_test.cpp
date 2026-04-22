#include "ears/factory.hpp"

#include <algorithm>
#include <gtest/gtest.h>
#include <memory>
#include <stdexcept>
#include <vector>

#include "ears/config.hpp"

#ifndef EARS_EXPECT_HAS_ONNX
#define EARS_EXPECT_HAS_ONNX 0
#endif
#ifndef EARS_EXPECT_HAS_TENSORRT
#define EARS_EXPECT_HAS_TENSORRT 0
#endif
#ifndef EARS_EXPECT_HAS_OPENVINO
#define EARS_EXPECT_HAS_OPENVINO 0
#endif
#ifndef EARS_EXPECT_HAS_COREML
#define EARS_EXPECT_HAS_COREML 0
#endif
#ifndef EARS_EXPECT_HAS_QNN
#define EARS_EXPECT_HAS_QNN 0
#endif

namespace ears {
namespace {

class TestVad final : public IVoiceActivityDetector {
public:
  float get_speech_probability(float const* /*samples*/, size_t /*num_samples*/) override {
    return 0.42f;
  }
};

class TestAsr final : public IAutomaticSpeechRecognizer {
public:
  AsrResult recognize(float const* /*samples*/, size_t /*num_samples*/) override {
    AsrResult result;
    result.text = "custom-asr";
    result.json = "{\"engine\":\"custom\"}";
    result.confidence = 0.42f;
    return result;
  }
};

class TestRuntimeAsr final : public IAutomaticSpeechRecognizer {
public:
  AsrResult recognize(float const* /*samples*/, size_t /*num_samples*/) override {
    AsrResult result;
    result.text = "runtime-asr";
    result.json = "{\"engine\":\"runtime\"}";
    result.confidence = 0.91f;
    return result;
  }
};

class TestProviderAsr final : public IAutomaticSpeechRecognizer {
public:
  AsrResult recognize(float const* /*samples*/, size_t /*num_samples*/) override {
    AsrResult result;
    result.text = "provider-asr";
    result.json = "{\"engine\":\"provider\"}";
    result.confidence = 0.77f;
    return result;
  }
};

class TestModelAsr final : public IAutomaticSpeechRecognizer {
public:
  AsrResult recognize(float const* /*samples*/, size_t /*num_samples*/) override {
    AsrResult result;
    result.text = "model-asr";
    result.json = "{\"engine\":\"model\"}";
    result.confidence = 0.63f;
    return result;
  }
};

class TestFallbackAsr final : public IAutomaticSpeechRecognizer {
public:
  AsrResult recognize(float const* /*samples*/, size_t /*num_samples*/) override {
    AsrResult result;
    result.text = "fallback-asr";
    result.json = "{\"engine\":\"fallback\"}";
    result.confidence = 0.11f;
    return result;
  }
};

class TestLlm final : public ISemanticRectifier {
public:
  std::string rectify(std::string const& phonetic_text, std::string const& /*app_context*/,
                      std::string const& /*history*/) override {
    return "cleaned:" + phonetic_text;
  }
};

TEST(FactoryTest, CreateVad_ReturnsNonNull) {
  Config config;
  auto vad = create_vad(config);
  EXPECT_NE(vad, nullptr);
}

TEST(FactoryTest, CreateVad_ReturnsValidVad) {
  Config config;
  auto vad = create_vad(config);

  std::vector<float> samples(512, 0.0f);
  float prob = vad->get_speech_probability(samples.data(), samples.size());
  EXPECT_GE(prob, 0.0f);
  EXPECT_LE(prob, 1.0f);
}

TEST(FactoryTest, CreateVad_DummyReturnsFixedProbability) {
  Config config;
  auto vad = create_vad(config);

  std::vector<float> samples(512, 0.0f);
  float prob = vad->get_speech_probability(samples.data(), samples.size());
  EXPECT_FLOAT_EQ(prob, 1.0f);  // DummyVad returns 1.0
}

TEST(FactoryTest, CreateAsr_ReturnsNonNull) {
  Config config;
  auto asr = create_asr(config);
  EXPECT_NE(asr, nullptr);
}

TEST(FactoryTest, CreateAsr_ReturnsValidAsr) {
  Config config;
  auto asr = create_asr(config);

  std::vector<float> samples(8000, 0.0f);
  AsrResult result = asr->recognize(samples.data(), samples.size());

  EXPECT_FALSE(result.text.empty());
  EXPECT_GE(result.confidence, 0.0f);
  EXPECT_LE(result.confidence, 1.0f);
}

TEST(FactoryTest, CreateAsr_UnknownModelReturnsNonEmptyResult) {
  Config config;
  config.asr.model = "unknown";
  config.asr.runtime.clear();
  auto asr = create_asr(config);

  std::vector<float> samples(1234, 0.0f);
  AsrResult result = asr->recognize(samples.data(), samples.size());

  EXPECT_FALSE(result.text.empty());
  EXPECT_GE(result.confidence, 0.0f);
}

TEST(FactoryTest, CreateLlm_ReturnsNonNull) {
  Config config;
  auto llm = create_llm(config);
  EXPECT_NE(llm, nullptr);
}

TEST(FactoryTest, CreateLlm_ReturnsValidLlm) {
  Config config;
  auto llm = create_llm(config);

  std::string output = llm->rectify("hello world", "VS Code", "previous context");
  EXPECT_EQ(output, "hello world");  // DummyLlm pass-through
}

TEST(FactoryTest, CreateLlm_DummyPassesThroughInput) {
  Config config;
  auto llm = create_llm(config);

  std::string input = "phonetic text with typos";
  std::string output = llm->rectify(input, "Chrome", "history");
  EXPECT_EQ(output, input);
}

TEST(FactoryTest, CreateAsr_ConfigMoonshine_ReturnsNonNull) {
  Config config;
  config.asr.model = "moonshine";
  config.asr.path = "models/moonshine.onnx";
  auto asr = create_asr(config);
  EXPECT_NE(asr, nullptr);
}

TEST(FactoryTest, CreateAsr_ConfigWhisper_ReturnsNonNull) {
  Config config;
  config.asr.model = "whisper";
  config.asr.path = "models/whisper.onnx";
  auto asr = create_asr(config);
  EXPECT_NE(asr, nullptr);
}

TEST(FactoryTest, CreateAsr_ConfigUnknown_ReturnsDummy) {
  Config config;
  config.asr.model = "unknown";
  auto asr = create_asr(config);
  EXPECT_NE(asr, nullptr);
  AsrResult r = asr->recognize(nullptr, 0);
  EXPECT_FALSE(r.text.empty());
  EXPECT_GE(r.confidence, 0.0f);
}

TEST(FactoryTest, CreateLlm_ConfigSmollm2_ReturnsNonNull) {
  Config config;
  config.llm.model = "smollm2";
  config.llm.path = "models/smollm2_360m.onnx";
  auto llm = create_llm(config);
  EXPECT_NE(llm, nullptr);
  std::string out = llm->rectify("test", "", "");
  EXPECT_EQ(out, "test");  // pass-through when using Dummy or stub
}

TEST(FactoryTest, CreateLlm_ConfigUnknown_ReturnsDummy) {
  Config config;
  config.llm.model = "unknown";
  auto llm = create_llm(config);
  EXPECT_NE(llm, nullptr);
  std::string out = llm->rectify("hello", "", "");
  EXPECT_EQ(out, "hello");
}

TEST(FactoryTest, RegisterCustomAdapters_UsedByCreateFunctions) {
  reset_factory_registry();
  Config config;

  EXPECT_TRUE(
      register_vad_factory("test-vad", [](Config const&) { return std::make_unique<TestVad>(); }));
  EXPECT_TRUE(
      register_asr_factory("test-asr", [](Config const&) { return std::make_unique<TestAsr>(); }));
  EXPECT_TRUE(
      register_llm_factory("test-llm", [](Config const&) { return std::make_unique<TestLlm>(); }));

  config.vad.model = "test-vad";
  config.asr.model = "test-asr";
  config.asr.runtime.clear();
  config.asr.family.clear();
  config.llm.enabled = true;
  config.llm.model = "test-llm";

  auto vad = create_vad(config);
  auto asr = create_asr(config);
  auto llm = create_llm(config);

  std::vector<float> samples(256, 0.0f);
  EXPECT_FLOAT_EQ(vad->get_speech_probability(samples.data(), samples.size()), 0.42f);

  AsrResult asr_result = asr->recognize(samples.data(), samples.size());
  EXPECT_EQ(asr_result.text, "custom-asr");
  EXPECT_EQ(asr_result.confidence, 0.42f);

  EXPECT_EQ(llm->rectify("raw", "", ""), "cleaned:raw");
}

TEST(FactoryTest, RegistryListIncludesBuiltinsAndCustom) {
  reset_factory_registry();

  EXPECT_TRUE(register_asr_factory("test-asr-list",
                                   [](Config const&) { return std::make_unique<TestAsr>(); }));

  std::vector<std::string> ids = list_registered_asr_factories();
  EXPECT_TRUE(std::find(ids.begin(), ids.end(), "dummy") != ids.end());
  EXPECT_TRUE(std::find(ids.begin(), ids.end(), "test-asr-list") != ids.end());
}

TEST(FactoryTest, BackendRegistrationTrace_ContainsStructuredEntries) {
  reset_factory_registry();
  std::vector<std::string> trace = get_backend_registration_trace();
  ASSERT_FALSE(trace.empty());

  bool has_structured_entry = false;
  for (std::string const& entry : trace) {
    if (entry.find("backend_register component=") != std::string::npos &&
        entry.find(" key=") != std::string::npos && entry.find(" source=") != std::string::npos) {
      has_structured_entry = true;
      break;
    }
  }
  EXPECT_TRUE(has_structured_entry);
}

TEST(FactoryTest, RuntimeFactoryRegistrationAndSelection) {
  reset_factory_registry();

  EXPECT_TRUE(register_asr_runtime_factory(
      "ctc", "onnx", [](Config const&) { return std::make_unique<TestRuntimeAsr>(); }));

  Config config;
  config.asr.model = "some-ctc-model";
  config.asr.family = "ctc";
  config.asr.runtime = "onnx";

  auto asr = create_asr(config);
  ASSERT_NE(asr, nullptr);
  AsrResult result = asr->recognize(nullptr, 0);
  EXPECT_EQ(result.text, "runtime-asr");
  EXPECT_TRUE(has_asr_runtime_family_capability("ctc", "onnx_runtime"));

  std::vector<std::string> runtime_ids = list_registered_asr_runtime_factories();
  EXPECT_TRUE(std::find(runtime_ids.begin(), runtime_ids.end(), "ctc::onnx_runtime") !=
              runtime_ids.end());
}

TEST(FactoryTest, ProviderFactoryRegistrationAndSelection) {
  reset_factory_registry();

  EXPECT_TRUE(register_asr_provider_factory("ctc", "onnx_runtime", "cuda", [](Config const&) {
    return std::make_unique<TestProviderAsr>();
  }));

  Config config;
  config.asr.model = "some-ctc-model";
  config.asr.family = "ctc";
  config.asr.runtime = "onnx_runtime";
  config.provider = "cuda";

  auto asr = create_asr(config);
  ASSERT_NE(asr, nullptr);
  AsrResult result = asr->recognize(nullptr, 0);
  EXPECT_EQ(result.text, "provider-asr");
  EXPECT_TRUE(has_asr_runtime_provider_capability("ctc", "onnx_runtime", "cuda"));

  std::vector<std::string> provider_ids = list_registered_asr_provider_factories();
  EXPECT_TRUE(std::find(provider_ids.begin(), provider_ids.end(), "ctc::onnx_runtime::cuda") !=
              provider_ids.end());
}

TEST(FactoryTest, CapabilityRuntimeQuery_MatchesCompileFlags) {
  std::vector<std::string> compiled = list_compiled_asr_runtimes();

  auto const has_entry = [&compiled](std::string const& runtime) {
    return std::find(compiled.begin(), compiled.end(), runtime) != compiled.end();
  };

  EXPECT_EQ(has_asr_runtime_capability("onnx_runtime"), EARS_EXPECT_HAS_ONNX != 0);
  EXPECT_EQ(has_asr_runtime_capability("onnx"), EARS_EXPECT_HAS_ONNX != 0);
  EXPECT_FALSE(has_asr_runtime_capability("tensorrt"));
  EXPECT_FALSE(has_asr_runtime_capability("openvino"));
  EXPECT_FALSE(has_asr_runtime_capability("coreml"));
  EXPECT_FALSE(has_asr_runtime_capability("qnn"));
  EXPECT_FALSE(has_asr_runtime_capability("not_a_runtime"));

  EXPECT_EQ(has_entry("onnx_runtime"), EARS_EXPECT_HAS_ONNX != 0);
  EXPECT_FALSE(has_entry("tensorrt"));
  EXPECT_FALSE(has_entry("openvino"));
  EXPECT_FALSE(has_entry("coreml"));
  EXPECT_FALSE(has_entry("qnn"));
}

TEST(FactoryTest, CapabilityFamilyQuery_MatchesCompileFlags) {
  reset_factory_registry();

  EXPECT_EQ(has_asr_runtime_family_capability("ctc", "onnx_runtime"), EARS_EXPECT_HAS_ONNX != 0);
  EXPECT_FALSE(has_asr_runtime_family_capability("ctc", "tensorrt"));
  EXPECT_FALSE(has_asr_runtime_family_capability("ctc", "openvino"));
  EXPECT_FALSE(has_asr_runtime_family_capability("ctc", "coreml"));
  EXPECT_FALSE(has_asr_runtime_family_capability("ctc", "qnn"));
  EXPECT_FALSE(has_asr_runtime_family_capability("unknown_family", "onnx_runtime"));
  EXPECT_FALSE(has_asr_runtime_family_capability("ctc", "unknown_runtime"));
}

TEST(FactoryTest, CapabilityProviderQuery_MatchesCompileFlags) {
  reset_factory_registry();

  EXPECT_EQ(has_asr_runtime_provider_capability("ctc", "onnx_runtime", "cuda"),
            EARS_EXPECT_HAS_ONNX != 0);
  EXPECT_EQ(has_asr_runtime_provider_capability("ctc", "onnx_runtime", "directml"),
            EARS_EXPECT_HAS_ONNX != 0);
  EXPECT_FALSE(has_asr_runtime_provider_capability("ctc", "onnx_runtime", "qnn_ep"));

  EXPECT_FALSE(has_asr_runtime_provider_capability("ctc", "tensorrt", "cuda"));
  EXPECT_FALSE(has_asr_runtime_provider_capability("unknown_family", "onnx_runtime", "cuda"));
  EXPECT_FALSE(has_asr_runtime_provider_capability("ctc", "onnx_runtime", "unknown_provider"));
}

TEST(FactoryTest, CreateAsr_RuntimeUnavailable_ReturnsExplicitStatus) {
  std::vector<std::string> const known_runtimes = {"onnx_runtime", "tensorrt", "openvino", "coreml",
                                                   "qnn"};
  std::string unavailable_runtime;
  for (std::string const& runtime : known_runtimes) {
    if (!has_asr_runtime_capability(runtime)) {
      unavailable_runtime = runtime;
      break;
    }
  }
  if (unavailable_runtime.empty()) {
    GTEST_SKIP() << "No unavailable known runtime in this build profile.";
  }

  Config config;
  config.asr.model = "test-model";
  config.asr.family = "ctc";
  config.asr.runtime = unavailable_runtime;

  auto asr = create_asr(config);
  ASSERT_NE(asr, nullptr);
  AsrResult result = asr->recognize(nullptr, 0);
  EXPECT_TRUE(result.json.find("\"code\":\"runtime_unavailable\"") != std::string::npos);
  EXPECT_TRUE(result.json.find("\"runtime\":\"" + unavailable_runtime + "\"") != std::string::npos);
}

TEST(FactoryTest, CreateAsr_UnsupportedFamilyRuntimeCombo_ReturnsExplicitStatus) {
  std::vector<std::string> const compiled = list_compiled_asr_runtimes();
  std::string runtime;
  for (std::string const& candidate : compiled) {
    if (candidate == "onnx_runtime" || candidate == "tensorrt" || candidate == "openvino" ||
        candidate == "coreml" || candidate == "qnn") {
      runtime = candidate;
      break;
    }
  }
  if (runtime.empty()) {
    GTEST_SKIP() << "No compiled known runtime in this build profile.";
  }

  Config config;
  config.asr.model = "test-model";
  config.asr.family = "not_registered_family";
  config.asr.runtime = runtime;

  auto asr = create_asr(config);
  ASSERT_NE(asr, nullptr);
  AsrResult result = asr->recognize(nullptr, 0);
  EXPECT_TRUE(result.json.find("\"code\":\"unsupported_family_runtime\"") != std::string::npos);
  EXPECT_TRUE(result.json.find("\"runtime\":\"" + runtime + "\"") != std::string::npos);
  EXPECT_TRUE(result.json.find("\"family\":\"not_registered_family\"") != std::string::npos);
}

TEST(FactoryTest, SelectionPrecedence_ProviderBeatsRuntimeModelAndFallback) {
  reset_factory_registry();

  EXPECT_TRUE(register_asr_factory(
      "dummy", [](Config const&) { return std::make_unique<TestFallbackAsr>(); }));
  EXPECT_TRUE(register_asr_factory("precedence-model",
                                   [](Config const&) { return std::make_unique<TestModelAsr>(); }));
  EXPECT_TRUE(register_asr_runtime_factory(
      "ctc", "onnx_runtime", [](Config const&) { return std::make_unique<TestRuntimeAsr>(); }));
  EXPECT_TRUE(register_asr_provider_factory("ctc", "onnx_runtime", "cuda", [](Config const&) {
    return std::make_unique<TestProviderAsr>();
  }));

  Config config;
  config.asr.model = "precedence-model";
  config.asr.family = "ctc";
  config.asr.runtime = "onnx_runtime";
  config.provider = "cuda";

  auto asr = create_asr(config);
  ASSERT_NE(asr, nullptr);
  AsrResult result = asr->recognize(nullptr, 0);
  EXPECT_EQ(result.text, "provider-asr");
}

TEST(FactoryTest, SelectionPrecedence_RuntimeBeatsModelAndFallbackWhenProviderMissing) {
  reset_factory_registry();

  EXPECT_TRUE(register_asr_factory(
      "dummy", [](Config const&) { return std::make_unique<TestFallbackAsr>(); }));
  EXPECT_TRUE(register_asr_factory("precedence-model",
                                   [](Config const&) { return std::make_unique<TestModelAsr>(); }));
  EXPECT_TRUE(register_asr_runtime_factory(
      "ctc", "custom_runtime", [](Config const&) { return std::make_unique<TestRuntimeAsr>(); }));

  Config config;
  config.asr.model = "precedence-model";
  config.asr.family = "ctc";
  config.asr.runtime = "custom_runtime";
  config.provider = "cuda";

  auto asr = create_asr(config);
  ASSERT_NE(asr, nullptr);
  AsrResult result = asr->recognize(nullptr, 0);
  EXPECT_EQ(result.text, "runtime-asr");
}

TEST(FactoryTest, SelectionPrecedence_ModelBeatsFallbackWhenRuntimeAndProviderMissing) {
  reset_factory_registry();

  EXPECT_TRUE(register_asr_factory(
      "dummy", [](Config const&) { return std::make_unique<TestFallbackAsr>(); }));
  EXPECT_TRUE(register_asr_factory("precedence-model",
                                   [](Config const&) { return std::make_unique<TestModelAsr>(); }));

  Config config;
  config.asr.model = "precedence-model";
  config.asr.family = "custom_family";
  config.asr.runtime = "custom_runtime";
  config.provider = "cuda";

  auto asr = create_asr(config);
  ASSERT_NE(asr, nullptr);
  AsrResult result = asr->recognize(nullptr, 0);
  EXPECT_EQ(result.text, "model-asr");
}

TEST(FactoryTest, SelectionPrecedence_FallbackUsedWhenHigherPriorityFactoriesFail) {
  reset_factory_registry();

  EXPECT_TRUE(register_asr_factory(
      "dummy", [](Config const&) { return std::make_unique<TestFallbackAsr>(); }));
  EXPECT_TRUE(register_asr_factory(
      "precedence-model", [](Config const&) -> std::unique_ptr<IAutomaticSpeechRecognizer> {
        throw std::runtime_error("model failure");
      }));
  EXPECT_TRUE(register_asr_runtime_factory(
      "ctc", "onnx_runtime", [](Config const&) -> std::unique_ptr<IAutomaticSpeechRecognizer> {
        throw std::runtime_error("runtime failure");
      }));
  EXPECT_TRUE(register_asr_provider_factory(
      "ctc", "onnx_runtime", "cuda",
      [](Config const&) -> std::unique_ptr<IAutomaticSpeechRecognizer> { return nullptr; }));

  Config config;
  config.asr.model = "precedence-model";
  config.asr.family = "ctc";
  config.asr.runtime = "onnx_runtime";
  config.provider = "cuda";

  auto asr = create_asr(config);
  ASSERT_NE(asr, nullptr);
  AsrResult result = asr->recognize(nullptr, 0);
  EXPECT_EQ(result.text, "fallback-asr");
}

}  // namespace
}  // namespace ears
