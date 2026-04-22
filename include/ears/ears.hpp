#pragma once

#include <functional>
#include <memory>

#include "ears/config.hpp"
#include "ears/types.hpp"

namespace ears {

class EarsPipeline {
public:
  explicit EarsPipeline(Config const& config);
  ~EarsPipeline();

  EarsPipeline(EarsPipeline const&) = delete;
  EarsPipeline& operator=(EarsPipeline const&) = delete;

  // Callback for incremental results
  using ResultCallback = std::function<void(TranscriptionResult const&)>;
  void set_result_callback(ResultCallback cb);

  // Stream lifecycle API (canonical surface)
  StreamHandle create_stream(StreamMode mode, ContextProfile const& profile,
                             StreamScheduling const& scheduling);
  void start_stream(StreamHandle handle);
  EnqueueStatus push_audio(StreamHandle handle, AudioChunk const& chunk,
                           Context const& ctx = Context{});
  void flush_stream(StreamHandle handle);
  void end_stream(StreamHandle handle);

  // Profile and thermal controls
  void upsert_profile(ContextProfile const& profile);
  bool bind_profile(StreamHandle handle, std::string const& profile_id);
  void upsert_pipeline_profile(PipelineProfileSpec const& pipeline_profile);
  StreamHandle start_pipeline(std::string const& pipeline_id);
  bool switch_stream_pipeline(StreamHandle handle, std::string const& pipeline_id);
  bool set_stream_param(StreamHandle handle, std::string const& key, std::string const& value);
  std::string get_stream_param(StreamHandle handle, std::string const& key) const;
  StreamSummary get_stream_summary(StreamHandle handle) const;
  void set_thermal_hint(ThermalHint hint);

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace ears
