
#include "ears/ears.hpp"
#include "ears/factory.hpp"
#include "ears/internal/string_utils.hpp"

#if EARS_HAS_ONNX
#include "ears/vad/silero_vad.hpp"
#endif

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ears {

namespace {

void split_stability(std::string const& text, int hold_words, std::string& immediate_out,
                     std::string& held_out) {
  if (text.empty() || hold_words <= 0) {
    immediate_out = text;
    held_out.clear();
    return;
  }

  std::vector<std::string> words;
  std::istringstream iss(text);
  std::string word;
  while (iss >> word) {
    words.push_back(std::move(word));
  }

  if (static_cast<int>(words.size()) <= hold_words) {
    immediate_out.clear();
    held_out = text;
    return;
  }

  std::ostringstream immediate;
  std::ostringstream held;
  int split = static_cast<int>(words.size()) - hold_words;
  for (int i = 0; i < split; ++i) {
    if (i > 0) {
      immediate << ' ';
    }
    immediate << words[static_cast<size_t>(i)];
  }
  for (int i = split; i < static_cast<int>(words.size()); ++i) {
    if (i > split) {
      held << ' ';
    }
    held << words[static_cast<size_t>(i)];
  }

  immediate_out = immediate.str();
  held_out = held.str();
}

std::string mode_to_string(StreamMode mode) {
  if (mode == StreamMode::batch) {
    return "batch";
  }
  return "realtime";
}

float clamp_priority(float value) {
  if (value < 0.0f) {
    return 0.0f;
  }
  if (value > 1.0f) {
    return 1.0f;
  }
  return value;
}

int qos_rank(StreamQosClass qos_class) {
  switch (qos_class) {
    case StreamQosClass::latency_critical:
      return 0;
    case StreamQosClass::latency_sensitive:
      return 1;
    case StreamQosClass::thermal_constrained:
      return 2;
    case StreamQosClass::opportunistic_idle:
      return 3;
  }
  return 4;
}

size_t queue_limit_for(StreamQosClass qos_class) {
  switch (qos_class) {
    case StreamQosClass::latency_critical:
      return 64;
    case StreamQosClass::latency_sensitive:
      return 48;
    case StreamQosClass::thermal_constrained:
      return 24;
    case StreamQosClass::opportunistic_idle:
      return 12;
  }
  return 16;
}

int64_t steady_now_ns() {
  auto const now = std::chrono::steady_clock::now().time_since_epoch();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(now).count();
}

enum class ParamApplyKind {
  ignored,
  light,
  heavy,
  invalid,
};

ParamApplyKind apply_param_to_config(Config& config, std::string const& key,
                                     std::string const& value) {
  if (key == "provider") {
    config.provider = value;
    return ParamApplyKind::heavy;
  }

  if (key == "vad.model") {
    config.vad.model = value;
    return ParamApplyKind::heavy;
  }
  if (key == "vad.path") {
    config.vad.path = value;
    return ParamApplyKind::heavy;
  }
  if (key == "vad.threshold") {
    float parsed = 0.0f;
    if (!internal::parse_float_strict(value, parsed)) {
      return ParamApplyKind::invalid;
    }
    config.vad.threshold = parsed;
    return ParamApplyKind::light;
  }
  if (key == "vad.hangover_ms") {
    int parsed = 0;
    if (!internal::parse_int_strict(value, parsed) || parsed < 0) {
      return ParamApplyKind::invalid;
    }
    config.vad.hangover_ms = parsed;
    return ParamApplyKind::light;
  }

  if (key == "asr.model") {
    config.asr.model = value;
    return ParamApplyKind::heavy;
  }
  if (key == "asr.family") {
    config.asr.family = value;
    return ParamApplyKind::heavy;
  }
  if (key == "asr.runtime") {
    config.asr.runtime = value;
    return ParamApplyKind::heavy;
  }
  if (key == "asr.path") {
    config.asr.path = value;
    return ParamApplyKind::heavy;
  }
  if (key == "asr.chunk_ms") {
    int parsed = 0;
    if (!internal::parse_int_strict(value, parsed) || parsed <= 0) {
      return ParamApplyKind::invalid;
    }
    config.asr.chunk_ms = parsed;
    return ParamApplyKind::light;
  }
  if (key == "asr.beam_size") {
    int parsed = 0;
    if (!internal::parse_int_strict(value, parsed) || parsed <= 0) {
      return ParamApplyKind::invalid;
    }
    config.asr.beam_size = parsed;
    return ParamApplyKind::light;
  }

  if (key == "llm.enabled") {
    bool parsed = false;
    if (!internal::parse_bool_strict(value, parsed)) {
      return ParamApplyKind::invalid;
    }
    config.llm.enabled = parsed;
    return ParamApplyKind::heavy;
  }
  if (key == "llm.model") {
    config.llm.model = value;
    return ParamApplyKind::heavy;
  }
  if (key == "llm.path") {
    config.llm.path = value;
    return ParamApplyKind::heavy;
  }
  if (key == "llm.quantization") {
    config.llm.quantization = value;
    return ParamApplyKind::heavy;
  }

  if (key == "stability.hold_ms") {
    int parsed = 0;
    if (!internal::parse_int_strict(value, parsed) || parsed < 0) {
      return ParamApplyKind::invalid;
    }
    config.stability.hold_ms = parsed;
    return ParamApplyKind::light;
  }
  if (key == "stability.history_ms") {
    int parsed = 0;
    if (!internal::parse_int_strict(value, parsed) || parsed < 0) {
      return ParamApplyKind::invalid;
    }
    config.stability.history_ms = parsed;
    return ParamApplyKind::light;
  }
  if (key == "stability.hold_words") {
    int parsed = 0;
    if (!internal::parse_int_strict(value, parsed) || parsed < 0) {
      return ParamApplyKind::invalid;
    }
    config.stability.hold_words = parsed;
    return ParamApplyKind::light;
  }

  return ParamApplyKind::invalid;
}

float percentile_from_sorted(std::vector<float> const& values, float percentile) {
  if (values.empty()) {
    return 0.0f;
  }
  float clamped = std::max(0.0f, std::min(100.0f, percentile));
  float position = (clamped / 100.0f) * static_cast<float>(values.size() - 1);
  size_t lower = static_cast<size_t>(position);
  size_t upper = std::min(values.size() - 1, lower + 1);
  float fraction = position - static_cast<float>(lower);
  return values[lower] + (values[upper] - values[lower]) * fraction;
}

}  // namespace

class EarsPipeline::Impl {
public:
  static constexpr int64_t NS_PER_MS = 1'000'000;
  static constexpr int64_t HARD_FAIL_NO_DECODE_NS = 5'000 * NS_PER_MS;
  static constexpr size_t COMPLETED_SUMMARY_CACHE_LIMIT = 4096;

  struct FlushRequest {
    bool done = false;
  };

  enum class WorkKind {
    audio,
    flush,
  };

  struct WorkItem {
    WorkKind kind = WorkKind::audio;
    AudioChunk chunk;
    Context ctx;
    int64_t enqueue_steady_ns = 0;
    uint64_t enqueue_seq = 0;
    std::shared_ptr<FlushRequest> flush;
  };

  struct StreamState {
    StreamHandle handle = 0;
    StreamMode mode = StreamMode::realtime;
    ContextProfile profile;
    StreamScheduling scheduling;
    bool started = false;
    bool ended = false;
    bool in_decode = false;
    bool hard_failed = false;

    uint64_t sequence_no = 0;
    uint64_t chunks_seen = 0;
    uint64_t results_emitted = 0;
    uint64_t errors = 0;
    int64_t speech_positive_window_start_ns = 0;
    int64_t last_successful_decode_ns = 0;

    std::unordered_map<std::string, std::string> params;
    std::string history;
    std::string previous_held;
    int64_t previous_timestamp_ns = 0;

    Config active_config;
    Config pending_config;
    bool has_pending_heavy_reconfigure = false;

    std::unique_ptr<IVoiceActivityDetector> vad;
    std::unique_ptr<IAutomaticSpeechRecognizer> asr;
    std::unique_ptr<ISemanticRectifier> llm;

    std::deque<WorkItem> queue;
    std::vector<float> latency_samples_ms;
    std::mutex state_mu;
  };

  explicit Impl(Config const& config) : base_config_(config) {
    ContextProfile profile;
    profile.profile_id = "default";
    profiles_.emplace(profile.profile_id, profile);
    worker_ = std::thread([this]() { worker_loop(); });
  }

  ~Impl() {
    shutdown();
  }

  void set_result_callback(ResultCallback cb) {
    std::lock_guard<std::mutex> lock(callback_mu_);
    callback_ = std::move(cb);
  }

  StreamHandle create_stream(StreamMode mode, ContextProfile const& profile,
                             StreamScheduling const& scheduling) {
    ContextProfile effective_profile = profile;
    if (effective_profile.profile_id.empty()) {
      effective_profile.profile_id = "default";
    }

    StreamScheduling effective_scheduling = scheduling;
    effective_scheduling.priority_hint = clamp_priority(effective_scheduling.priority_hint);

    auto stream = std::make_shared<StreamState>();
    stream->mode = mode;
    stream->profile = effective_profile;
    stream->scheduling = effective_scheduling;
    stream->active_config = base_config_;

    {
      std::lock_guard<std::mutex> state_lock(stream->state_mu);
      (void)rebuild_components_locked(*stream, stream->active_config);
      (void)apply_profile_hints_locked(*stream, stream->profile, true);
    }

    std::lock_guard<std::mutex> lock(mu_);
    StreamHandle const handle = next_stream_handle_++;
    stream->handle = handle;
    streams_[handle] = stream;
    profiles_[effective_profile.profile_id] = effective_profile;
    return handle;
  }

  void start_stream(StreamHandle handle) {
    auto stream = get_stream(handle);
    if (!stream) {
      return;
    }

    std::lock_guard<std::mutex> state_lock(stream->state_mu);
    if (stream->ended) {
      return;
    }
    if (!stream->vad || !stream->asr || !stream->llm) {
      (void)rebuild_components_locked(*stream, stream->active_config);
    }
    apply_pending_reconfigure_if_any_locked(*stream);
    stream->hard_failed = false;
    stream->speech_positive_window_start_ns = 0;
    stream->last_successful_decode_ns = 0;
    stream->started = true;
  }
  EnqueueStatus push_audio(StreamHandle handle, AudioChunk const& chunk, Context const& ctx) {
    auto stream = get_stream(handle);
    if (!stream) {
      return EnqueueStatus::invalid_state;
    }

    if (chunk.samples.empty()) {
      return EnqueueStatus::ok;
    }

    std::lock_guard<std::mutex> lock(mu_);
    std::lock_guard<std::mutex> state_lock(stream->state_mu);
    if (!stream->started || stream->ended || stream->hard_failed) {
      return EnqueueStatus::invalid_state;
    }
    if (!can_run_under_thermal_locked(*stream)) {
      return EnqueueStatus::backpressure;
    }

    size_t const queue_limit = queue_limit_for(stream->scheduling.qos_class);
    if (stream->queue.size() >= queue_limit) {
      return EnqueueStatus::backpressure;
    }

    WorkItem item;
    item.kind = WorkKind::audio;
    item.chunk = chunk;
    item.ctx = ctx;
    item.enqueue_steady_ns = steady_now_ns();
    item.enqueue_seq = ++enqueue_sequence_;
    stream->queue.push_back(std::move(item));
    stream->chunks_seen += 1;
    pending_work_items_ += 1;

    cv_.notify_one();
    return EnqueueStatus::ok;
  }

  void flush_stream(StreamHandle handle) {
    auto stream = get_stream(handle);
    if (!stream) {
      return;
    }

    if (std::this_thread::get_id() == worker_thread_id_) {
      auto emissions = process_flush_inline(stream);
      dispatch_results(emissions);
      return;
    }

    std::shared_ptr<FlushRequest> request;
    {
      std::lock_guard<std::mutex> state_lock(stream->state_mu);
      if (!stream->started || stream->ended) {
        auto emissions = finalize_stream_locked(*stream);
        dispatch_results(emissions);
        return;
      }
    }

    std::unique_lock<std::mutex> lock(mu_);
    {
      std::lock_guard<std::mutex> state_lock(stream->state_mu);
      request = std::make_shared<FlushRequest>();
      WorkItem flush_item;
      flush_item.kind = WorkKind::flush;
      flush_item.enqueue_steady_ns = steady_now_ns();
      flush_item.enqueue_seq = ++enqueue_sequence_;
      flush_item.flush = request;
      stream->queue.push_back(std::move(flush_item));
      pending_work_items_ += 1;
    }

    cv_.notify_one();
    cv_.wait(lock, [&]() { return request->done || stop_worker_; });
  }

  void end_stream(StreamHandle handle) {
    flush_stream(handle);

    auto stream = get_stream(handle);
    if (!stream) {
      return;
    }

    std::vector<TranscriptionResult> emissions;
    {
      std::lock_guard<std::mutex> lock(mu_);
      std::lock_guard<std::mutex> state_lock(stream->state_mu);
      if (stream->ended) {
        return;
      }

      StreamSummary const summary = summarize_locked(*stream);
      emissions.push_back(
          make_result_locked(*stream, "", "", stream_summary_json(summary, stream->active_config),
                             steady_now_ns(), true, "stream_summary", "", 0.0f, 0.0f, 0.0f, 0.0f));
      store_completed_summary_locked(summary);

      stream->ended = true;
      stream->started = false;
      stream->queue.clear();
      stream->in_decode = false;
      stream->hard_failed = false;
      stream->speech_positive_window_start_ns = 0;
      stream->last_successful_decode_ns = 0;
      stream->history.clear();
      stream->previous_held.clear();
      stream->vad.reset();
      stream->asr.reset();
      stream->llm.reset();
      streams_.erase(handle);
    }

    dispatch_results(emissions);
    cv_.notify_all();
  }

  void upsert_profile(ContextProfile const& profile) {
    ContextProfile effective = profile;
    if (effective.profile_id.empty()) {
      effective.profile_id = "default";
    }

    std::lock_guard<std::mutex> lock(mu_);
    profiles_[effective.profile_id] = effective;
  }

  bool bind_profile(StreamHandle handle, std::string const& profile_id) {
    ContextProfile profile;
    auto stream = get_stream(handle);
    if (!stream) {
      return false;
    }

    {
      std::lock_guard<std::mutex> lock(mu_);
      auto it = profiles_.find(profile_id);
      if (it == profiles_.end()) {
        return false;
      }
      profile = it->second;
    }

    std::lock_guard<std::mutex> state_lock(stream->state_mu);
    if (stream->ended) {
      return false;
    }

    stream->profile = profile;
    bool const allow_immediate_heavy = !stream->started || stream->chunks_seen == 0;
    return apply_profile_hints_locked(*stream, stream->profile, allow_immediate_heavy);
  }

  void set_thermal_hint(ThermalHint hint) {
    std::lock_guard<std::mutex> lock(mu_);
    thermal_hint_ = hint;
    cv_.notify_all();
  }

  void upsert_pipeline_profile(PipelineProfileSpec const& pipeline_profile) {
    if (pipeline_profile.pipeline_id.empty()) {
      return;
    }

    PipelineProfileSpec effective = pipeline_profile;
    if (effective.profile.profile_id.empty()) {
      effective.profile.profile_id = "default";
    }
    effective.scheduling.priority_hint = clamp_priority(effective.scheduling.priority_hint);

    std::lock_guard<std::mutex> lock(mu_);
    profiles_[effective.profile.profile_id] = effective.profile;
    pipeline_profiles_[effective.pipeline_id] = effective;
  }

  StreamHandle start_pipeline(std::string const& pipeline_id) {
    PipelineProfileSpec pipeline;
    {
      std::lock_guard<std::mutex> lock(mu_);
      auto it = pipeline_profiles_.find(pipeline_id);
      if (it == pipeline_profiles_.end()) {
        return 0;
      }
      pipeline = it->second;
    }

    StreamHandle handle = create_stream(pipeline.mode, pipeline.profile, pipeline.scheduling);
    start_stream(handle);
    return handle;
  }
  bool switch_stream_pipeline(StreamHandle handle, std::string const& pipeline_id) {
    PipelineProfileSpec pipeline;
    {
      std::lock_guard<std::mutex> lock(mu_);
      auto it = pipeline_profiles_.find(pipeline_id);
      if (it == pipeline_profiles_.end()) {
        return false;
      }
      pipeline = it->second;
    }

    auto stream = get_stream(handle);
    if (!stream) {
      return false;
    }

    std::lock_guard<std::mutex> state_lock(stream->state_mu);
    if (stream->ended) {
      return false;
    }

    stream->mode = pipeline.mode;
    stream->profile = pipeline.profile;
    stream->scheduling = pipeline.scheduling;
    stream->scheduling.priority_hint = clamp_priority(stream->scheduling.priority_hint);

    bool const allow_immediate_heavy = !stream->started || stream->chunks_seen == 0;
    return apply_profile_hints_locked(*stream, stream->profile, allow_immediate_heavy);
  }

  bool set_stream_param(StreamHandle handle, std::string const& key, std::string const& value) {
    if (key.empty()) {
      return false;
    }

    auto stream = get_stream(handle);
    if (!stream) {
      return false;
    }

    std::lock_guard<std::mutex> state_lock(stream->state_mu);
    if (stream->ended) {
      return false;
    }

    bool const allow_immediate_heavy = !stream->started || stream->chunks_seen == 0;
    return apply_stream_param_locked(*stream, key, value, allow_immediate_heavy);
  }

  std::string get_stream_param(StreamHandle handle, std::string const& key) const {
    if (key.empty()) {
      return "";
    }

    auto stream = get_stream(handle);
    if (!stream) {
      return "";
    }

    std::lock_guard<std::mutex> state_lock(stream->state_mu);
    auto it = stream->params.find(key);
    if (it == stream->params.end()) {
      return "";
    }
    return it->second;
  }

  StreamSummary get_stream_summary(StreamHandle handle) const {
    auto stream = get_stream(handle);
    if (stream) {
      std::lock_guard<std::mutex> state_lock(stream->state_mu);
      return summarize_locked(*stream);
    }

    std::lock_guard<std::mutex> lock(mu_);
    auto completed_it = completed_summaries_.find(handle);
    if (completed_it != completed_summaries_.end()) {
      return completed_it->second;
    }
    return StreamSummary{};
  }

private:
  struct Candidate {
    std::shared_ptr<StreamState> stream;
    int lane_rank = 0;
    float priority_hint = 0.0f;
    uint64_t enqueue_seq = 0;
    bool is_flush = false;
  };

  std::shared_ptr<StreamState> get_stream(StreamHandle handle) const {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = streams_.find(handle);
    if (it == streams_.end()) {
      return nullptr;
    }
    return it->second;
  }

  bool is_better_candidate(Candidate const& candidate, Candidate const& best) const {
    // Formal precedence order: flush barrier > qos lane > priority_hint > FIFO.
    if (candidate.lane_rank != best.lane_rank) {
      return candidate.lane_rank < best.lane_rank;
    }
    if (candidate.priority_hint != best.priority_hint) {
      return candidate.priority_hint > best.priority_hint;
    }
    return candidate.enqueue_seq < best.enqueue_seq;
  }

  bool rebuild_components_locked(StreamState& stream, Config const& config) {
    try {
      stream.vad = create_vad(config);
      stream.asr = create_asr(config);
      stream.llm = create_llm(config);
      return stream.vad && stream.asr && stream.llm;
    } catch (...) {
      stream.errors += 1;
      return false;
    }
  }

  void apply_pending_reconfigure_if_any_locked(StreamState& stream) {
    if (!stream.has_pending_heavy_reconfigure) {
      return;
    }

    stream.active_config = stream.pending_config;
    (void)rebuild_components_locked(stream, stream.active_config);
    stream.has_pending_heavy_reconfigure = false;
  }
  bool apply_stream_param_locked(StreamState& stream, std::string const& key,
                                 std::string const& value, bool allow_immediate_heavy) {
    Config preview = stream.active_config;
    ParamApplyKind const kind = apply_param_to_config(preview, key, value);
    if (kind == ParamApplyKind::invalid) {
      return false;
    }

    if (kind == ParamApplyKind::ignored) {
      stream.params[key] = value;
      return true;
    }

    if (kind == ParamApplyKind::light) {
      stream.params[key] = value;
      stream.active_config = preview;
      if (stream.has_pending_heavy_reconfigure) {
        Config pending_preview = stream.pending_config;
        ParamApplyKind const pending_kind = apply_param_to_config(pending_preview, key, value);
        if (pending_kind != ParamApplyKind::invalid && pending_kind != ParamApplyKind::ignored) {
          stream.pending_config = pending_preview;
        }
      }
      return true;
    }

    if (allow_immediate_heavy) {
      stream.params[key] = value;
      stream.active_config = preview;
      (void)rebuild_components_locked(stream, stream.active_config);
      stream.has_pending_heavy_reconfigure = false;
      return true;
    }

    Config pending =
        stream.has_pending_heavy_reconfigure ? stream.pending_config : stream.active_config;
    ParamApplyKind const pending_kind = apply_param_to_config(pending, key, value);
    if (pending_kind == ParamApplyKind::invalid) {
      return false;
    }
    stream.params[key] = value;
    stream.pending_config = pending;
    stream.has_pending_heavy_reconfigure = true;
    return true;
  }

  bool apply_profile_hints_locked(StreamState& stream, ContextProfile const& profile,
                                  bool allow_immediate_heavy) {
    bool all_ok = true;
    for (auto const& hint : profile.hints) {
      if (!apply_stream_param_locked(stream, hint.first, hint.second, allow_immediate_heavy)) {
        all_ok = false;
        stream.errors += 1;
      }
    }
    return all_ok;
  }

  bool can_run_under_thermal_locked(StreamState const& stream) const {
    if (thermal_hint_ == ThermalHint::normal) {
      return true;
    }

    if (thermal_hint_ == ThermalHint::reduce_gpu) {
      return stream.scheduling.qos_class != StreamQosClass::opportunistic_idle;
    }

    return stream.scheduling.qos_class == StreamQosClass::latency_critical;
  }

  StreamSummary summarize_locked(StreamState const& stream) const {
    StreamSummary summary;
    summary.stream_id = stream.handle;
    summary.chunks_seen = stream.chunks_seen;
    summary.results_emitted = stream.results_emitted;
    summary.errors = stream.errors;
    if (stream.latency_samples_ms.empty()) {
      return summary;
    }

    std::vector<float> sorted = stream.latency_samples_ms;
    std::sort(sorted.begin(), sorted.end());
    float sum = 0.0f;
    for (float value : sorted) {
      sum += value;
    }
    summary.p50_end_to_end_ms = percentile_from_sorted(sorted, 50.0f);
    summary.p95_end_to_end_ms = percentile_from_sorted(sorted, 95.0f);
    summary.p99_end_to_end_ms = percentile_from_sorted(sorted, 99.0f);
    summary.avg_end_to_end_ms = sum / static_cast<float>(sorted.size());
    summary.peak_end_to_end_ms = sorted.back();
    return summary;
  }

  void store_completed_summary_locked(StreamSummary const& summary) {
    completed_summaries_[summary.stream_id] = summary;
    completed_summary_order_.push_back(summary.stream_id);
    while (completed_summary_order_.size() > COMPLETED_SUMMARY_CACHE_LIMIT) {
      StreamHandle oldest = completed_summary_order_.front();
      completed_summary_order_.pop_front();
      completed_summaries_.erase(oldest);
    }
  }

  std::string stream_summary_json(StreamSummary const& summary, Config const& config) const {
    std::string const config_checksum = effective_config_checksum_hex(config);
    std::ostringstream out;
    out << "{\"event\":\"stream_summary\""
        << ",\"stream_id\":" << summary.stream_id << ",\"chunks_seen\":" << summary.chunks_seen
        << ",\"results_emitted\":" << summary.results_emitted << ",\"errors\":" << summary.errors
        << ",\"p50_end_to_end_ms\":" << summary.p50_end_to_end_ms
        << ",\"p95_end_to_end_ms\":" << summary.p95_end_to_end_ms
        << ",\"p99_end_to_end_ms\":" << summary.p99_end_to_end_ms
        << ",\"avg_end_to_end_ms\":" << summary.avg_end_to_end_ms
        << ",\"peak_end_to_end_ms\":" << summary.peak_end_to_end_ms << ",\"config_checksum\":\""
        << config_checksum << "\"}";
    return out.str();
  }

  int64_t activity_time_ns_for(WorkItem const& item) const {
    if (item.chunk.timestamp_ns > 0) {
      return item.chunk.timestamp_ns;
    }
    if (item.enqueue_steady_ns > 0) {
      return item.enqueue_steady_ns;
    }
    return steady_now_ns();
  }

  bool maybe_hard_fail_no_decode_locked(StreamState& stream, int64_t activity_ns,
                                        int64_t result_timestamp_ns,
                                        std::vector<TranscriptionResult>& emissions) {
    if (stream.scheduling.qos_class != StreamQosClass::latency_critical) {
      return false;
    }
    if (stream.speech_positive_window_start_ns <= 0) {
      stream.speech_positive_window_start_ns = activity_ns;
    }

    int64_t const reference_ns = stream.last_successful_decode_ns > 0
                                     ? stream.last_successful_decode_ns
                                     : stream.speech_positive_window_start_ns;
    if ((activity_ns - reference_ns) < HARD_FAIL_NO_DECODE_NS) {
      return false;
    }

    stream.hard_failed = true;
    stream.started = false;
    stream.errors += 1;
    emissions.push_back(make_result_locked(
        stream, "", "", "{}", result_timestamp_ns, false, "error",
        "hard_fail_no_decode_output_5s_restart_required", 0.0f, 0.0f, 0.0f, 0.0f));
    return true;
  }

  int64_t hold_ns_for_locked(StreamState const& stream) const {
    return static_cast<int64_t>(stream.active_config.stability.hold_ms) * NS_PER_MS;
  }

  int hold_words_for_locked(StreamState const& stream) const {
    return std::max(0, stream.active_config.stability.hold_words);
  }

  void reset_vad_state_locked(StreamState& stream) {
#if EARS_HAS_ONNX
    if (auto* silero = dynamic_cast<SileroVad*>(stream.vad.get())) {
      silero->reset();
    }
#else
    (void)stream;
#endif
  }

  TranscriptionResult make_result_locked(StreamState& stream, std::string const& corrected,
                                         std::string const& raw_text, std::string const& raw_json,
                                         int64_t timestamp_ns, bool is_final,
                                         std::string const& status,
                                         std::string const& error_message, float confidence,
                                         float decode_ms, float queue_delay_ms,
                                         float end_to_end_ms) {
    TranscriptionResult result;
    result.stream_id = stream.handle;
    result.sequence_no = ++stream.sequence_no;
    result.corrected_text = corrected;
    result.raw_asr_text = raw_text;
    result.raw_asr_json = raw_json;
    result.timestamp_ns = timestamp_ns;
    result.start_ms = timestamp_ns / NS_PER_MS;
    result.end_ms = result.start_ms;
    result.is_final = is_final;
    result.mode = mode_to_string(stream.mode);
    result.status = status;
    result.error_message = error_message;
    result.profile_id = stream.profile.profile_id;
    result.profile_version = stream.profile.version;
    result.confidence = confidence;
    result.model_id = stream.active_config.asr.model;
    result.provider = stream.active_config.provider;
    result.decode_ms = decode_ms;
    result.queue_delay_ms = queue_delay_ms;
    result.end_to_end_ms = end_to_end_ms;
    stream.results_emitted += 1;

    if (end_to_end_ms > 0.0f) {
      stream.latency_samples_ms.push_back(end_to_end_ms);
    }
    return result;
  }

  std::vector<TranscriptionResult> finalize_stream_locked(StreamState& stream) {
    std::vector<TranscriptionResult> emissions;

    reset_vad_state_locked(stream);
    if (!stream.previous_held.empty()) {
      emissions.push_back(make_result_locked(stream, stream.previous_held, stream.previous_held,
                                             "{}", stream.previous_timestamp_ns, true, "ok", "",
                                             0.0f, 0.0f, 0.0f, 0.0f));
      stream.previous_held.clear();
    }

    stream.history.clear();
    stream.speech_positive_window_start_ns = 0;
    stream.last_successful_decode_ns = 0;
    apply_pending_reconfigure_if_any_locked(stream);
    return emissions;
  }

  std::vector<TranscriptionResult> process_flush_inline(
      std::shared_ptr<StreamState> const& stream) {
    std::lock_guard<std::mutex> state_lock(stream->state_mu);
    return finalize_stream_locked(*stream);
  }
  std::vector<TranscriptionResult> process_audio_item(std::shared_ptr<StreamState> const& stream,
                                                      WorkItem const& item) {
    std::vector<TranscriptionResult> emissions;

    std::lock_guard<std::mutex> state_lock(stream->state_mu);
    if (stream->ended || !stream->started) {
      return emissions;
    }

    if (!stream->previous_held.empty() &&
        (item.chunk.timestamp_ns - stream->previous_timestamp_ns) >= hold_ns_for_locked(*stream)) {
      emissions.push_back(make_result_locked(*stream, stream->previous_held, stream->previous_held,
                                             "{}", stream->previous_timestamp_ns, true, "ok", "",
                                             0.0f, 0.0f, 0.0f, 0.0f));
      stream->previous_held.clear();
    }

    if (!stream->vad || !stream->asr || !stream->llm) {
      if (!rebuild_components_locked(*stream, stream->active_config)) {
        emissions.push_back(make_result_locked(*stream, "", "", "{}", item.chunk.timestamp_ns,
                                               false, "error", "component_rebuild_failed", 0.0f,
                                               0.0f, 0.0f, 0.0f));
        stream->errors += 1;
        return emissions;
      }
    }

    float speech_prob =
        stream->vad->get_speech_probability(item.chunk.samples.data(), item.chunk.samples.size());
    int64_t const activity_ns = activity_time_ns_for(item);
    if (speech_prob < stream->active_config.vad.threshold) {
      stream->speech_positive_window_start_ns = 0;
      stream->last_successful_decode_ns = 0;
      return emissions;
    }
    if (stream->speech_positive_window_start_ns <= 0) {
      stream->speech_positive_window_start_ns = activity_ns;
    }

    int64_t const decode_begin_ns = steady_now_ns();
    float const queue_delay_ms =
        std::max(0.0f, static_cast<float>(decode_begin_ns - item.enqueue_steady_ns) / NS_PER_MS);

    AsrResult asr_out;
    try {
      asr_out = stream->asr->recognize(item.chunk.samples.data(), item.chunk.samples.size());
    } catch (std::exception const& e) {
      emissions.push_back(make_result_locked(*stream, "", "", "{}", item.chunk.timestamp_ns, false,
                                             "error", e.what(), 0.0f, 0.0f, queue_delay_ms,
                                             queue_delay_ms));
      stream->errors += 1;
      (void)maybe_hard_fail_no_decode_locked(*stream, activity_ns, item.chunk.timestamp_ns,
                                             emissions);
      return emissions;
    } catch (...) {
      emissions.push_back(make_result_locked(*stream, "", "", "{}", item.chunk.timestamp_ns, false,
                                             "error", "asr_decode_failed", 0.0f, 0.0f,
                                             queue_delay_ms, queue_delay_ms));
      stream->errors += 1;
      (void)maybe_hard_fail_no_decode_locked(*stream, activity_ns, item.chunk.timestamp_ns,
                                             emissions);
      return emissions;
    }

    if (!asr_out.text.empty()) {
      stream->last_successful_decode_ns = activity_ns;
    } else {
      (void)maybe_hard_fail_no_decode_locked(*stream, activity_ns, item.chunk.timestamp_ns,
                                             emissions);
      if (stream->hard_failed) {
        return emissions;
      }
    }

    int64_t const decode_end_ns = steady_now_ns();
    float const decode_ms =
        static_cast<float>(decode_end_ns - decode_begin_ns) / static_cast<float>(NS_PER_MS);
    float const end_to_end_ms = queue_delay_ms + decode_ms;

    std::string corrected = asr_out.text;
    if (stream->active_config.llm.enabled) {
      corrected = stream->llm->rectify(asr_out.text, item.ctx.app_window_title, stream->history);
      stream->history = corrected;
      if (stream->history.size() > 200) {
        size_t pos = stream->history.find_last_of(" \t", stream->history.size() - 100);
        if (pos != std::string::npos) {
          stream->history = stream->history.substr(pos + 1);
        }
      }
    } else {
      stream->history.clear();
    }

    std::string immediate;
    std::string held;
    split_stability(corrected, hold_words_for_locked(*stream), immediate, held);
    stream->previous_held = held;
    stream->previous_timestamp_ns = item.chunk.timestamp_ns;

    if (!immediate.empty()) {
      emissions.push_back(make_result_locked(
          *stream, immediate, asr_out.text, asr_out.json, item.chunk.timestamp_ns, held.empty(),
          "ok", "", asr_out.confidence, decode_ms, queue_delay_ms, end_to_end_ms));
    }

    return emissions;
  }

  void dispatch_results(std::vector<TranscriptionResult> const& emissions) {
    if (emissions.empty()) {
      return;
    }

    ResultCallback callback;
    {
      std::lock_guard<std::mutex> lock(callback_mu_);
      callback = callback_;
    }
    if (!callback) {
      return;
    }

    for (auto const& result : emissions) {
      callback(result);
    }
  }

  bool pick_next_work_locked(std::shared_ptr<StreamState>& out_stream, WorkItem& out_item) {
    Candidate best;
    bool found = false;

    for (auto const& entry : streams_) {
      auto const& stream = entry.second;
      std::lock_guard<std::mutex> state_lock(stream->state_mu);
      if (!stream->started || stream->ended || stream->in_decode || stream->queue.empty()) {
        continue;
      }

      WorkItem const& front = stream->queue.front();
      bool const is_flush = front.kind == WorkKind::flush;
      if (!is_flush && !can_run_under_thermal_locked(*stream)) {
        continue;
      }

      Candidate candidate;
      candidate.stream = stream;
      candidate.is_flush = is_flush;
      candidate.lane_rank = is_flush ? -1 : qos_rank(stream->scheduling.qos_class);
      candidate.priority_hint = stream->scheduling.priority_hint;
      candidate.enqueue_seq = front.enqueue_seq;

      if (!found || is_better_candidate(candidate, best)) {
        best = candidate;
        found = true;
      }
    }
    if (!found || !best.stream) {
      return false;
    }

    {
      std::lock_guard<std::mutex> state_lock(best.stream->state_mu);
      if (best.stream->queue.empty() || best.stream->in_decode) {
        return false;
      }
      out_item = std::move(best.stream->queue.front());
      best.stream->queue.pop_front();
      best.stream->in_decode = true;
    }

    if (pending_work_items_ > 0) {
      pending_work_items_ -= 1;
    }

    out_stream = best.stream;
    return true;
  }

  void worker_loop() {
    worker_thread_id_ = std::this_thread::get_id();

    while (true) {
      std::shared_ptr<StreamState> stream;
      WorkItem item;

      {
        std::unique_lock<std::mutex> lock(mu_);
        cv_.wait(lock, [&]() { return stop_worker_ || pending_work_items_ > 0; });
        if (stop_worker_) {
          return;
        }

        if (!pick_next_work_locked(stream, item)) {
          continue;
        }
      }

      std::vector<TranscriptionResult> emissions;
      if (item.kind == WorkKind::audio) {
        emissions = process_audio_item(stream, item);
      } else {
        emissions = process_flush_inline(stream);
      }

      dispatch_results(emissions);

      {
        std::lock_guard<std::mutex> lock(mu_);
        {
          std::lock_guard<std::mutex> state_lock(stream->state_mu);
          stream->in_decode = false;
          if (stream->hard_failed && !stream->queue.empty()) {
            size_t const dropped = stream->queue.size();
            for (auto const& queued_item : stream->queue) {
              if (queued_item.kind == WorkKind::flush && queued_item.flush) {
                queued_item.flush->done = true;
              }
            }
            stream->queue.clear();
            if (pending_work_items_ > dropped) {
              pending_work_items_ -= dropped;
            } else {
              pending_work_items_ = 0;
            }
          }
        }

        if (item.kind == WorkKind::flush && item.flush) {
          item.flush->done = true;
        }
      }

      cv_.notify_all();
    }
  }

  void shutdown() {
    {
      std::lock_guard<std::mutex> lock(mu_);
      if (stop_worker_) {
        return;
      }
      stop_worker_ = true;
    }

    cv_.notify_all();
    if (worker_.joinable()) {
      worker_.join();
    }
  }

  Config base_config_;

  mutable std::mutex mu_;
  mutable std::mutex callback_mu_;
  std::condition_variable cv_;
  bool stop_worker_ = false;
  std::thread worker_;
  std::thread::id worker_thread_id_{};

  ThermalHint thermal_hint_ = ThermalHint::normal;
  StreamHandle next_stream_handle_ = 1;
  uint64_t enqueue_sequence_ = 0;
  size_t pending_work_items_ = 0;

  std::unordered_map<StreamHandle, std::shared_ptr<StreamState>> streams_;
  std::unordered_map<StreamHandle, StreamSummary> completed_summaries_;
  std::deque<StreamHandle> completed_summary_order_;
  std::unordered_map<std::string, ContextProfile> profiles_;
  std::unordered_map<std::string, PipelineProfileSpec> pipeline_profiles_;

  ResultCallback callback_;
};

EarsPipeline::EarsPipeline(Config const& config) : impl_(std::make_unique<Impl>(config)) {}

EarsPipeline::~EarsPipeline() = default;

void EarsPipeline::set_result_callback(ResultCallback cb) {
  impl_->set_result_callback(std::move(cb));
}

StreamHandle EarsPipeline::create_stream(StreamMode mode, ContextProfile const& profile,
                                         StreamScheduling const& scheduling) {
  return impl_->create_stream(mode, profile, scheduling);
}

void EarsPipeline::start_stream(StreamHandle handle) {
  impl_->start_stream(handle);
}

EnqueueStatus EarsPipeline::push_audio(StreamHandle handle, AudioChunk const& chunk,
                                       Context const& ctx) {
  return impl_->push_audio(handle, chunk, ctx);
}
void EarsPipeline::flush_stream(StreamHandle handle) {
  impl_->flush_stream(handle);
}

void EarsPipeline::end_stream(StreamHandle handle) {
  impl_->end_stream(handle);
}

void EarsPipeline::upsert_profile(ContextProfile const& profile) {
  impl_->upsert_profile(profile);
}

bool EarsPipeline::bind_profile(StreamHandle handle, std::string const& profile_id) {
  return impl_->bind_profile(handle, profile_id);
}

void EarsPipeline::upsert_pipeline_profile(PipelineProfileSpec const& pipeline_profile) {
  impl_->upsert_pipeline_profile(pipeline_profile);
}

StreamHandle EarsPipeline::start_pipeline(std::string const& pipeline_id) {
  return impl_->start_pipeline(pipeline_id);
}

bool EarsPipeline::switch_stream_pipeline(StreamHandle handle, std::string const& pipeline_id) {
  return impl_->switch_stream_pipeline(handle, pipeline_id);
}

bool EarsPipeline::set_stream_param(StreamHandle handle, std::string const& key,
                                    std::string const& value) {
  return impl_->set_stream_param(handle, key, value);
}

std::string EarsPipeline::get_stream_param(StreamHandle handle, std::string const& key) const {
  return impl_->get_stream_param(handle, key);
}

StreamSummary EarsPipeline::get_stream_summary(StreamHandle handle) const {
  return impl_->get_stream_summary(handle);
}

void EarsPipeline::set_thermal_hint(ThermalHint hint) {
  impl_->set_thermal_hint(hint);
}

}  // namespace ears
