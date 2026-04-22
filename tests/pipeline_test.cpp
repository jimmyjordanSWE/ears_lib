#include <algorithm>
#include <chrono>
#include <gtest/gtest.h>
#include <memory>
#include <stdexcept>
#include <vector>

#include "ears/ears.hpp"
#include "ears/factory.hpp"

namespace ears {
namespace {

class AlwaysFailAsr final : public IAutomaticSpeechRecognizer {
public:
  AsrResult recognize(float const* /*samples*/, size_t /*num_samples*/) override {
    throw std::runtime_error("forced_decode_failure");
  }
};

TEST(PipelineTest, Construct_DoesNotThrow) {
  Config config;
  EXPECT_NO_THROW(EarsPipeline pipeline(config));
}

StreamHandle create_started_stream(EarsPipeline& pipeline,
                                   StreamQosClass qos = StreamQosClass::latency_critical,
                                   char const* profile_id = "test") {
  ContextProfile profile;
  profile.profile_id = profile_id;
  StreamScheduling scheduling;
  scheduling.qos_class = qos;
  StreamHandle stream = pipeline.create_stream(StreamMode::realtime, profile, scheduling);
  pipeline.start_stream(stream);
  return stream;
}

AudioChunk make_audio_chunk(size_t num_samples = 8000, int64_t timestamp_ns = 1'000'000'000LL) {
  AudioChunk chunk;
  chunk.samples.resize(num_samples, 0.0f);
  chunk.timestamp_ns = timestamp_ns;
  chunk.tier = "tier1";
  return chunk;
}

TEST(PipelineTest, PushAudio_EmptyChunk_DoesNotCallCallback) {
  Config config;
  EarsPipeline pipeline(config);

  int transcript_count = 0;
  pipeline.set_result_callback([&transcript_count](TranscriptionResult const& result) {
    if (result.status == "stream_summary" || result.status == "error") {
      return;
    }
    ++transcript_count;
  });

  StreamHandle stream = create_started_stream(pipeline);

  AudioChunk chunk;
  chunk.samples = {};
  chunk.timestamp_ns = 0;
  chunk.tier = "tier1";

  Context ctx;
  ctx.app_window_title = "Test";

  (void)pipeline.push_audio(stream, chunk, ctx);
  pipeline.flush_stream(stream);
  pipeline.end_stream(stream);

  EXPECT_EQ(transcript_count, 0);
}

TEST(PipelineTest, PushAudio_SpeechChunk_CallsCallback) {
  Config config;
  EarsPipeline pipeline(config);

  std::vector<TranscriptionResult> results;
  auto callback = [&results](TranscriptionResult const& r) { results.push_back(r); };
  pipeline.set_result_callback(callback);

  StreamHandle stream = create_started_stream(pipeline);
  AudioChunk chunk =
      make_audio_chunk(8000, std::chrono::steady_clock::now().time_since_epoch().count());

  Context ctx;
  ctx.app_window_title = "Test";

  EXPECT_EQ(pipeline.push_audio(stream, chunk, ctx), EnqueueStatus::ok);
  pipeline.flush_stream(stream);
  pipeline.end_stream(stream);

  EXPECT_GE(results.size(), 1u);
  EXPECT_FALSE(results[0].corrected_text.empty());
  EXPECT_FALSE(results[0].raw_asr_text.empty());
}

TEST(PipelineTest, PushAudio_StabilitySplit_EmitsImmediateAndHeld) {
  Config config;
  EarsPipeline pipeline(config);

  std::vector<TranscriptionResult> results;
  pipeline.set_result_callback([&results](TranscriptionResult const& r) { results.push_back(r); });

  StreamHandle stream = create_started_stream(pipeline);
  AudioChunk chunk = make_audio_chunk();

  Context ctx;

  EXPECT_EQ(pipeline.push_audio(stream, chunk, ctx), EnqueueStatus::ok);
  pipeline.flush_stream(stream);
  pipeline.end_stream(stream);

  EXPECT_GE(results.size(), 1u);
  EXPECT_EQ(results[0].timestamp_ns, 1000000000);
}

TEST(PipelineTest, FlushStream_EmitsHeldText) {
  Config config;
  EarsPipeline pipeline(config);

  std::vector<TranscriptionResult> results;
  pipeline.set_result_callback([&results](TranscriptionResult const& r) { results.push_back(r); });

  StreamHandle stream = create_started_stream(pipeline);
  AudioChunk chunk = make_audio_chunk();

  Context ctx;
  EXPECT_EQ(pipeline.push_audio(stream, chunk, ctx), EnqueueStatus::ok);

  size_t count_before_flush = results.size();
  pipeline.flush_stream(stream);
  pipeline.end_stream(stream);
  size_t count_after_flush = results.size();

  EXPECT_GE(count_after_flush, count_before_flush);
}

TEST(PipelineTest, PushAudio_VadBelowThreshold_DoesNotEmit) {
  Config config;
  config.vad.threshold = 1.5f;  // Impossible to exceed

  EarsPipeline pipeline(config);

  std::vector<TranscriptionResult> results;
  pipeline.set_result_callback([&results](TranscriptionResult const& r) { results.push_back(r); });

  StreamHandle stream = create_started_stream(pipeline);
  AudioChunk chunk = make_audio_chunk();

  Context ctx;
  EXPECT_EQ(pipeline.push_audio(stream, chunk, ctx), EnqueueStatus::ok);
  pipeline.flush_stream(stream);
  pipeline.end_stream(stream);

  size_t transcript_count = 0;
  for (TranscriptionResult const& result : results) {
    if (result.status != "stream_summary" && result.status != "error") {
      ++transcript_count;
    }
  }
  EXPECT_EQ(transcript_count, 0u);
}

TEST(PipelineTest, PushAudio_PassesAppContextToLlm) {
  Config config;
  EarsPipeline pipeline(config);

  std::vector<TranscriptionResult> results;
  pipeline.set_result_callback([&results](TranscriptionResult const& r) { results.push_back(r); });

  StreamHandle stream = create_started_stream(pipeline);
  AudioChunk chunk = make_audio_chunk();

  Context ctx;
  ctx.app_window_title = "Visual Studio Code";
  ctx.process_name = "code.exe";

  EXPECT_EQ(pipeline.push_audio(stream, chunk, ctx), EnqueueStatus::ok);
  pipeline.flush_stream(stream);
  pipeline.end_stream(stream);

  EXPECT_GE(results.size(), 1u);
  EXPECT_FALSE(results[0].raw_asr_text.empty());
}

TEST(PipelineTest, SetResultCallback_ReplacesPrevious) {
  Config config;
  EarsPipeline pipeline(config);
  StreamHandle stream = create_started_stream(pipeline);

  int call_count_1 = 0;
  pipeline.set_result_callback([&call_count_1](TranscriptionResult const&) { ++call_count_1; });

  AudioChunk chunk = make_audio_chunk();
  Context ctx;

  EXPECT_EQ(pipeline.push_audio(stream, chunk, ctx), EnqueueStatus::ok);
  pipeline.flush_stream(stream);
  int count_after_first = call_count_1;

  int call_count_2 = 0;
  pipeline.set_result_callback([&call_count_2](TranscriptionResult const&) { ++call_count_2; });
  EXPECT_EQ(pipeline.push_audio(stream, chunk, ctx), EnqueueStatus::ok);
  pipeline.flush_stream(stream);
  pipeline.end_stream(stream);

  EXPECT_EQ(call_count_1, count_after_first);
  EXPECT_GE(call_count_2, 1);
}

TEST(PipelineTest, PushAudio_WithoutStart_ReturnsInvalidState) {
  Config config;
  EarsPipeline pipeline(config);

  ContextProfile profile;
  profile.profile_id = "test";
  StreamScheduling scheduling;
  StreamHandle stream = pipeline.create_stream(StreamMode::realtime, profile, scheduling);

  AudioChunk chunk;
  chunk.samples.resize(1600, 0.0f);
  chunk.timestamp_ns = 1000000000;

  EnqueueStatus status = pipeline.push_audio(stream, chunk);
  EXPECT_EQ(status, EnqueueStatus::invalid_state);
}

TEST(PipelineTest, StreamLifecycle_CreateStartPushFlushEnd) {
  Config config;
  EarsPipeline pipeline(config);

  std::vector<TranscriptionResult> results;
  pipeline.set_result_callback([&results](TranscriptionResult const& r) { results.push_back(r); });

  ContextProfile profile;
  profile.profile_id = "default";
  StreamScheduling scheduling;
  scheduling.qos_class = StreamQosClass::latency_critical;
  StreamHandle stream = pipeline.create_stream(StreamMode::realtime, profile, scheduling);
  pipeline.start_stream(stream);

  AudioChunk chunk;
  chunk.samples.resize(8000, 0.0f);
  chunk.timestamp_ns = 1000000000;
  chunk.tier = "tier1";

  EnqueueStatus status = pipeline.push_audio(stream, chunk);
  EXPECT_EQ(status, EnqueueStatus::ok);
  pipeline.flush_stream(stream);
  EXPECT_GE(results.size(), 1u);
  EXPECT_EQ(results[0].stream_id, stream);
  EXPECT_EQ(results[0].sequence_no, 1u);
  EXPECT_EQ(results[0].mode, "realtime");

  pipeline.end_stream(stream);

  status = pipeline.push_audio(stream, chunk);
  EXPECT_EQ(status, EnqueueStatus::invalid_state);
}

TEST(PipelineTest, BindProfile_AppliesProfileMetadataToResults) {
  Config config;
  EarsPipeline pipeline(config);

  std::vector<TranscriptionResult> results;
  pipeline.set_result_callback([&results](TranscriptionResult const& r) { results.push_back(r); });

  ContextProfile default_profile;
  default_profile.profile_id = "default";

  ContextProfile code_profile;
  code_profile.profile_id = "code";
  code_profile.version = 3;
  pipeline.upsert_profile(code_profile);

  StreamScheduling scheduling;
  StreamHandle stream = pipeline.create_stream(StreamMode::realtime, default_profile, scheduling);
  pipeline.start_stream(stream);
  EXPECT_TRUE(pipeline.bind_profile(stream, "code"));

  AudioChunk chunk;
  chunk.samples.resize(8000, 0.0f);
  chunk.timestamp_ns = 1000000000;
  chunk.tier = "tier1";
  EXPECT_EQ(pipeline.push_audio(stream, chunk), EnqueueStatus::ok);
  pipeline.flush_stream(stream);
  ASSERT_GE(results.size(), 1u);
  EXPECT_EQ(results[0].profile_id, "code");
  EXPECT_EQ(results[0].profile_version, 3);
}

TEST(PipelineTest, CreateStream_ReturnsUniqueHandles) {
  Config config;
  EarsPipeline pipeline(config);

  StreamHandle first = create_started_stream(pipeline, StreamQosClass::latency_critical, "first");
  StreamHandle second = create_started_stream(pipeline, StreamQosClass::latency_critical, "second");
  EXPECT_NE(first, 0u);
  EXPECT_NE(second, 0u);
  EXPECT_NE(first, second);

  AudioChunk chunk = make_audio_chunk(4000);
  EXPECT_EQ(pipeline.push_audio(first, chunk), EnqueueStatus::ok);
  EXPECT_EQ(pipeline.push_audio(second, chunk), EnqueueStatus::ok);
  pipeline.flush_stream(first);
  pipeline.flush_stream(second);
  pipeline.end_stream(first);
  pipeline.end_stream(second);
}

TEST(PipelineTest, ThermalHint_MinimumGpuBlocksNonCriticalStreams) {
  Config config;
  EarsPipeline pipeline(config);

  ContextProfile profile;
  profile.profile_id = "batch";
  StreamScheduling scheduling;
  scheduling.qos_class = StreamQosClass::latency_sensitive;
  StreamHandle stream = pipeline.create_stream(StreamMode::batch, profile, scheduling);
  pipeline.start_stream(stream);

  pipeline.set_thermal_hint(ThermalHint::minimum_gpu);

  AudioChunk chunk;
  chunk.samples.resize(4000, 0.0f);
  chunk.timestamp_ns = 1000000000;

  EXPECT_EQ(pipeline.push_audio(stream, chunk), EnqueueStatus::backpressure);
}

TEST(PipelineTest, ThermalHint_MinimumGpuAllowsLatencyCritical) {
  Config config;
  EarsPipeline pipeline(config);

  ContextProfile profile;
  profile.profile_id = "critical";
  StreamScheduling scheduling;
  scheduling.qos_class = StreamQosClass::latency_critical;
  StreamHandle stream = pipeline.create_stream(StreamMode::realtime, profile, scheduling);
  pipeline.start_stream(stream);

  pipeline.set_thermal_hint(ThermalHint::minimum_gpu);

  AudioChunk chunk;
  chunk.samples.resize(4000, 0.0f);
  chunk.timestamp_ns = 1000000000;

  EXPECT_EQ(pipeline.push_audio(stream, chunk), EnqueueStatus::ok);
}

TEST(PipelineTest, PipelineProfile_StartAndSwitchStream) {
  Config config;
  EarsPipeline pipeline(config);

  std::vector<TranscriptionResult> results;
  pipeline.set_result_callback([&results](TranscriptionResult const& r) { results.push_back(r); });

  PipelineProfileSpec batch_profile;
  batch_profile.pipeline_id = "batch_polite";
  batch_profile.mode = StreamMode::batch;
  batch_profile.profile.profile_id = "batch";
  batch_profile.profile.version = 2;
  batch_profile.scheduling.qos_class = StreamQosClass::latency_sensitive;
  pipeline.upsert_pipeline_profile(batch_profile);

  StreamHandle stream = pipeline.start_pipeline("batch_polite");
  EXPECT_NE(stream, 0u);

  AudioChunk chunk;
  chunk.samples.resize(4000, 0.0f);
  chunk.timestamp_ns = 1000000000;
  EXPECT_EQ(pipeline.push_audio(stream, chunk), EnqueueStatus::ok);
  pipeline.flush_stream(stream);
  ASSERT_GE(results.size(), 1u);
  EXPECT_EQ(results[0].mode, "batch");
  EXPECT_EQ(results[0].profile_id, "batch");
  EXPECT_EQ(results[0].profile_version, 2);

  PipelineProfileSpec realtime_profile;
  realtime_profile.pipeline_id = "realtime_dictation";
  realtime_profile.mode = StreamMode::realtime;
  realtime_profile.profile.profile_id = "default";
  realtime_profile.scheduling.qos_class = StreamQosClass::latency_critical;
  pipeline.upsert_pipeline_profile(realtime_profile);
  EXPECT_TRUE(pipeline.switch_stream_pipeline(stream, "realtime_dictation"));

  results.clear();
  chunk.timestamp_ns += 500000000;
  EXPECT_EQ(pipeline.push_audio(stream, chunk), EnqueueStatus::ok);
  pipeline.flush_stream(stream);
  ASSERT_GE(results.size(), 1u);
  EXPECT_EQ(results[0].mode, "realtime");
  EXPECT_EQ(results[0].profile_id, "default");
}

TEST(PipelineTest, StartPipeline_MissingProfileReturnsZero) {
  Config config;
  EarsPipeline pipeline(config);
  EXPECT_EQ(pipeline.start_pipeline("does_not_exist"), 0u);
}

TEST(PipelineTest, StreamParams_SetAndGet) {
  Config config;
  EarsPipeline pipeline(config);

  ContextProfile profile;
  profile.profile_id = "default";
  StreamScheduling scheduling;
  StreamHandle stream = pipeline.create_stream(StreamMode::realtime, profile, scheduling);
  pipeline.start_stream(stream);

  EXPECT_TRUE(pipeline.set_stream_param(stream, "asr.chunk_ms", "320"));
  EXPECT_TRUE(pipeline.set_stream_param(stream, "asr.beam_size", "4"));
  EXPECT_TRUE(pipeline.set_stream_param(stream, "stability.hold_words", "2"));
  EXPECT_EQ(pipeline.get_stream_param(stream, "asr.chunk_ms"), "320");
  EXPECT_EQ(pipeline.get_stream_param(stream, "asr.beam_size"), "4");
  EXPECT_EQ(pipeline.get_stream_param(stream, "stability.hold_words"), "2");
  EXPECT_EQ(pipeline.get_stream_param(stream, "missing"), "");

  pipeline.end_stream(stream);
  EXPECT_FALSE(pipeline.set_stream_param(stream, "asr.chunk_ms", "480"));
}

TEST(PipelineTest, StreamSummary_HasCountsAndLatencyStats) {
  Config config;
  EarsPipeline pipeline(config);

  ContextProfile profile;
  profile.profile_id = "default";
  StreamScheduling scheduling;
  StreamHandle stream = pipeline.create_stream(StreamMode::realtime, profile, scheduling);
  pipeline.start_stream(stream);

  AudioChunk chunk;
  chunk.samples.resize(4000, 0.0f);
  chunk.timestamp_ns = 1000000000;
  EXPECT_EQ(pipeline.push_audio(stream, chunk), EnqueueStatus::ok);
  pipeline.flush_stream(stream);

  StreamSummary summary = pipeline.get_stream_summary(stream);
  EXPECT_EQ(summary.stream_id, stream);
  EXPECT_EQ(summary.chunks_seen, 1u);
  EXPECT_GE(summary.results_emitted, 1u);
  EXPECT_GE(summary.p50_end_to_end_ms, 0.0f);
  EXPECT_GE(summary.p95_end_to_end_ms, summary.p50_end_to_end_ms);
  EXPECT_GE(summary.p99_end_to_end_ms, summary.p95_end_to_end_ms);
  EXPECT_GE(summary.peak_end_to_end_ms, summary.p99_end_to_end_ms);
}

TEST(PipelineTest, EndStream_EmitsStreamSummaryEvent) {
  Config config;
  EarsPipeline pipeline(config);

  std::vector<TranscriptionResult> results;
  pipeline.set_result_callback([&results](TranscriptionResult const& r) { results.push_back(r); });

  ContextProfile profile;
  profile.profile_id = "default";
  StreamScheduling scheduling;
  StreamHandle stream = pipeline.create_stream(StreamMode::realtime, profile, scheduling);
  pipeline.start_stream(stream);

  AudioChunk chunk;
  chunk.samples.resize(4000, 0.0f);
  chunk.timestamp_ns = 1000000000;
  EXPECT_EQ(pipeline.push_audio(stream, chunk), EnqueueStatus::ok);
  pipeline.flush_stream(stream);
  pipeline.end_stream(stream);

  auto summary_it = std::find_if(
      results.begin(), results.end(),
      [](TranscriptionResult const& result) { return result.status == "stream_summary"; });
  ASSERT_NE(summary_it, results.end());
  EXPECT_EQ(summary_it->stream_id, stream);
  EXPECT_TRUE(summary_it->is_final);
  EXPECT_NE(summary_it->raw_asr_json.find("\"event\":\"stream_summary\""), std::string::npos);
  EXPECT_NE(summary_it->raw_asr_json.find("\"config_checksum\":\""), std::string::npos);
}

TEST(PipelineTest, StreamParam_LightweightAppliesOnNextChunk) {
  Config config;
  EarsPipeline pipeline(config);

  std::vector<TranscriptionResult> results;
  pipeline.set_result_callback([&results](TranscriptionResult const& r) { results.push_back(r); });

  ContextProfile profile;
  profile.profile_id = "default";
  StreamScheduling scheduling;
  StreamHandle stream = pipeline.create_stream(StreamMode::realtime, profile, scheduling);
  pipeline.start_stream(stream);

  AudioChunk chunk;
  chunk.samples.resize(8000, 0.0f);
  chunk.timestamp_ns = 1000000000;

  EXPECT_EQ(pipeline.push_audio(stream, chunk), EnqueueStatus::ok);
  pipeline.flush_stream(stream);
  ASSERT_GE(results.size(), 1u);

  results.clear();
  EXPECT_TRUE(pipeline.set_stream_param(stream, "vad.threshold", "1.5"));

  chunk.timestamp_ns += 10'000'000;  // +10 ms, below hold window so no held flush result.
  EXPECT_EQ(pipeline.push_audio(stream, chunk), EnqueueStatus::ok);
  pipeline.flush_stream(stream);
  EXPECT_EQ(results.size(), 0u);
}

TEST(PipelineTest, StreamParam_HeavyDeferredUntilFlushBoundary) {
  Config config;
  EarsPipeline pipeline(config);

  std::vector<TranscriptionResult> results;
  pipeline.set_result_callback([&results](TranscriptionResult const& r) { results.push_back(r); });

  ContextProfile profile;
  profile.profile_id = "default";
  StreamScheduling scheduling;
  StreamHandle stream = pipeline.create_stream(StreamMode::realtime, profile, scheduling);
  pipeline.start_stream(stream);

  AudioChunk chunk;
  chunk.samples.resize(8000, 0.0f);
  chunk.timestamp_ns = 1000000000;

  EXPECT_EQ(pipeline.push_audio(stream, chunk), EnqueueStatus::ok);
  pipeline.flush_stream(stream);
  ASSERT_GE(results.size(), 1u);
  std::string first_model = results.back().model_id;

  EXPECT_TRUE(pipeline.set_stream_param(stream, "asr.model", "whisper"));

  results.clear();
  chunk.timestamp_ns += 300'000'000;  // +300 ms
  EXPECT_EQ(pipeline.push_audio(stream, chunk), EnqueueStatus::ok);
  pipeline.flush_stream(stream);
  ASSERT_GE(results.size(), 1u);
  EXPECT_EQ(results.back().model_id, first_model);

  results.clear();
  chunk.timestamp_ns += 300'000'000;  // +300 ms
  EXPECT_EQ(pipeline.push_audio(stream, chunk), EnqueueStatus::ok);
  pipeline.flush_stream(stream);
  ASSERT_GE(results.size(), 1u);
  EXPECT_EQ(results.back().model_id, "whisper");
}

TEST(PipelineTest, StreamParam_InvalidValueRejectedAndNotPersisted) {
  Config config;
  EarsPipeline pipeline(config);

  ContextProfile profile;
  profile.profile_id = "default";
  StreamScheduling scheduling;
  StreamHandle stream = pipeline.create_stream(StreamMode::realtime, profile, scheduling);
  pipeline.start_stream(stream);

  EXPECT_FALSE(pipeline.set_stream_param(stream, "asr.chunk_ms", "not-a-number"));
  EXPECT_FALSE(pipeline.set_stream_param(stream, "asr.unknown_knob", "123"));
  EXPECT_EQ(pipeline.get_stream_param(stream, "asr.chunk_ms"), "");
  EXPECT_EQ(pipeline.get_stream_param(stream, "asr.unknown_knob"), "");
}

TEST(PipelineTest, BindProfile_HeavyHintsDeferredUntilFlushBoundary) {
  Config config;
  EarsPipeline pipeline(config);

  std::vector<TranscriptionResult> results;
  pipeline.set_result_callback([&results](TranscriptionResult const& r) { results.push_back(r); });

  ContextProfile initial;
  initial.profile_id = "initial";
  StreamScheduling scheduling;
  StreamHandle stream = pipeline.create_stream(StreamMode::realtime, initial, scheduling);
  pipeline.start_stream(stream);

  AudioChunk chunk;
  chunk.samples.resize(8000, 0.0f);
  chunk.timestamp_ns = 1000000000;
  EXPECT_EQ(pipeline.push_audio(stream, chunk), EnqueueStatus::ok);
  pipeline.flush_stream(stream);
  ASSERT_GE(results.size(), 1u);
  std::string first_model = results.back().model_id;

  ContextProfile heavy_profile;
  heavy_profile.profile_id = "heavy";
  heavy_profile.hints["asr.model"] = "whisper";
  pipeline.upsert_profile(heavy_profile);
  EXPECT_TRUE(pipeline.bind_profile(stream, "heavy"));

  results.clear();
  chunk.timestamp_ns += 300000000;
  EXPECT_EQ(pipeline.push_audio(stream, chunk), EnqueueStatus::ok);
  pipeline.flush_stream(stream);
  ASSERT_GE(results.size(), 1u);
  EXPECT_EQ(results.back().model_id, first_model);

  results.clear();
  chunk.timestamp_ns += 300000000;
  EXPECT_EQ(pipeline.push_audio(stream, chunk), EnqueueStatus::ok);
  pipeline.flush_stream(stream);
  ASSERT_GE(results.size(), 1u);
  EXPECT_EQ(results.back().model_id, "whisper");
}

TEST(PipelineTest, BindProfile_LightHintsApplyOnNextChunk) {
  Config config;
  EarsPipeline pipeline(config);

  std::vector<TranscriptionResult> results;
  pipeline.set_result_callback([&results](TranscriptionResult const& r) { results.push_back(r); });

  ContextProfile initial;
  initial.profile_id = "initial";
  StreamScheduling scheduling;
  StreamHandle stream = pipeline.create_stream(StreamMode::realtime, initial, scheduling);
  pipeline.start_stream(stream);

  AudioChunk chunk;
  chunk.samples.resize(8000, 0.0f);
  chunk.timestamp_ns = 1000000000;
  EXPECT_EQ(pipeline.push_audio(stream, chunk), EnqueueStatus::ok);
  pipeline.flush_stream(stream);
  ASSERT_GE(results.size(), 1u);

  ContextProfile light_profile;
  light_profile.profile_id = "light";
  light_profile.hints["vad.threshold"] = "1.5";
  pipeline.upsert_profile(light_profile);
  EXPECT_TRUE(pipeline.bind_profile(stream, "light"));

  results.clear();
  chunk.timestamp_ns += 10000000;  // 10 ms; avoid held-token timeout emission.
  EXPECT_EQ(pipeline.push_audio(stream, chunk), EnqueueStatus::ok);
  pipeline.flush_stream(stream);
  EXPECT_EQ(results.size(), 0u);
}

TEST(PipelineTest, LatencyCriticalHardFailsAfterFiveSecondsNoDecodeAndNeedsRestart) {
  reset_factory_registry();
  EXPECT_TRUE(register_asr_factory(
      "always-fail-hardfail", [](Config const&) { return std::make_unique<AlwaysFailAsr>(); }));

  Config config;
  config.vad.model = "dummy";
  config.vad.threshold = 0.0f;
  config.asr.model = "always-fail-hardfail";
  config.asr.runtime.clear();
  config.asr.family.clear();
  {
    auto asr = create_asr(config);
    ASSERT_NE(asr, nullptr);
    EXPECT_THROW(asr->recognize(nullptr, 0), std::runtime_error);
  }

  EarsPipeline pipeline(config);
  std::vector<TranscriptionResult> results;
  pipeline.set_result_callback([&results](TranscriptionResult const& r) { results.push_back(r); });

  ContextProfile profile;
  profile.profile_id = "default";
  StreamScheduling scheduling;
  scheduling.qos_class = StreamQosClass::latency_critical;
  StreamHandle stream = pipeline.create_stream(StreamMode::realtime, profile, scheduling);
  pipeline.start_stream(stream);

  AudioChunk chunk;
  chunk.samples.resize(4000, 0.0f);
  chunk.tier = "tier1";

  for (int i = 0; i < 6; ++i) {
    chunk.timestamp_ns = 1'000'000'000LL + static_cast<int64_t>(i) * 1'000'000'000LL;
    EXPECT_EQ(pipeline.push_audio(stream, chunk), EnqueueStatus::ok);
  }
  pipeline.flush_stream(stream);

  chunk.timestamp_ns = 7'000'000'000LL;
  EXPECT_EQ(pipeline.push_audio(stream, chunk), EnqueueStatus::invalid_state);

  auto hard_fail_it =
      std::find_if(results.begin(), results.end(), [](TranscriptionResult const& result) {
        return result.error_message == "hard_fail_no_decode_output_5s_restart_required";
      });
  ASSERT_NE(hard_fail_it, results.end());

  pipeline.start_stream(stream);
  EXPECT_EQ(pipeline.push_audio(stream, chunk), EnqueueStatus::ok);
  pipeline.flush_stream(stream);
}

TEST(PipelineTest, NonCriticalStreamStaysFailSoftUnderContinuousDecodeErrors) {
  reset_factory_registry();
  EXPECT_TRUE(register_asr_factory(
      "always-fail-soft", [](Config const&) { return std::make_unique<AlwaysFailAsr>(); }));

  Config config;
  config.vad.model = "dummy";
  config.vad.threshold = 0.0f;
  config.asr.model = "always-fail-soft";
  config.asr.runtime.clear();
  config.asr.family.clear();

  EarsPipeline pipeline(config);
  std::vector<TranscriptionResult> results;
  pipeline.set_result_callback([&results](TranscriptionResult const& r) { results.push_back(r); });

  ContextProfile profile;
  profile.profile_id = "default";
  StreamScheduling scheduling;
  scheduling.qos_class = StreamQosClass::latency_sensitive;
  StreamHandle stream = pipeline.create_stream(StreamMode::realtime, profile, scheduling);
  pipeline.start_stream(stream);

  AudioChunk chunk;
  chunk.samples.resize(4000, 0.0f);
  chunk.tier = "tier1";

  for (int i = 0; i < 7; ++i) {
    chunk.timestamp_ns = 1'000'000'000LL + static_cast<int64_t>(i) * 1'000'000'000LL;
    EXPECT_EQ(pipeline.push_audio(stream, chunk), EnqueueStatus::ok);
    pipeline.flush_stream(stream);
  }

  chunk.timestamp_ns = 8'000'000'000LL;
  EXPECT_EQ(pipeline.push_audio(stream, chunk), EnqueueStatus::ok);
  pipeline.flush_stream(stream);

  auto hard_fail_it =
      std::find_if(results.begin(), results.end(), [](TranscriptionResult const& result) {
        return result.error_message == "hard_fail_no_decode_output_5s_restart_required";
      });
  EXPECT_EQ(hard_fail_it, results.end());
}

}  // namespace
}  // namespace ears
