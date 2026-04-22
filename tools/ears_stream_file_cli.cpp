#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include "ears/ears.hpp"

namespace {

bool read_f32le_file(std::string const& path, std::vector<float>& out_samples) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    return false;
  }

  input.seekg(0, std::ios::end);
  std::streamoff const bytes = input.tellg();
  if (bytes <= 0 || (bytes % static_cast<std::streamoff>(sizeof(float))) != 0) {
    return false;
  }
  input.seekg(0, std::ios::beg);

  out_samples.resize(static_cast<size_t>(bytes / static_cast<std::streamoff>(sizeof(float))));
  input.read(reinterpret_cast<char*>(out_samples.data()), bytes);
  return static_cast<bool>(input);
}

std::string safe_text(std::string text) {
  for (char& ch : text) {
    if (ch == '\r' || ch == '\n') {
      ch = ' ';
    }
  }
  return text;
}

}  // namespace

int main(int argc, char** argv) {
  std::cout.setf(std::ios::unitbuf);
  std::cerr.setf(std::ios::unitbuf);
  if (argc < 3) {
    std::cerr << "usage: ears_stream_file_cli <audio_f32le> <out_txt> [provider]\n";
    return 1;
  }

  std::string const audio_path = argv[1];
  std::string const out_path = argv[2];
  std::string const provider = argc >= 4 ? argv[3] : "cuda";

  std::vector<float> samples;
  if (!read_f32le_file(audio_path, samples)) {
    std::cerr << "failed to read audio file: " << audio_path << "\n";
    return 2;
  }
  std::cout << "loaded audio: " << audio_path << " samples=" << samples.size() << "\n";

  std::ofstream out(out_path, std::ios::trunc);
  if (!out) {
    std::cerr << "failed to open output file: " << out_path << "\n";
    return 3;
  }
  std::cout << "writing callback log: " << out_path << "\n";

  ears::Config config;
  config.provider = provider;
  std::cout << "creating pipeline provider=" << provider << "\n";

  ears::EarsPipeline pipeline(config);
  pipeline.set_result_callback([&out](ears::TranscriptionResult const& result) {
    std::string const line =
        "seq=" + std::to_string(result.sequence_no) + " status=" + result.status + " final=" +
        (result.is_final ? "true" : "false") + " provider=" + result.provider + " text=" +
        safe_text(result.corrected_text);
    out << line;
    if (!result.error_message.empty()) {
      out << " error=" << safe_text(result.error_message);
    }
    out << "\n";
    out.flush();
    std::cout << line;
    if (!result.error_message.empty()) {
      std::cout << " error=" << safe_text(result.error_message);
    }
    std::cout << "\n";
  });

  ears::ContextProfile profile;
  profile.profile_id = "stream-test";
  ears::StreamScheduling scheduling;
  scheduling.qos_class = ears::StreamQosClass::latency_critical;

  ears::StreamHandle const stream =
      pipeline.create_stream(ears::StreamMode::realtime, profile, scheduling);
  std::cout << "created stream id=" << stream << "\n";
  pipeline.start_stream(stream);
  std::cout << "started stream\n";

  constexpr size_t kChunkSamples = 1600;  // 100 ms at 16 kHz
  int64_t timestamp_ns = 0;
  for (size_t offset = 0; offset < samples.size(); offset += kChunkSamples) {
    size_t const count = std::min(kChunkSamples, samples.size() - offset);
    ears::AudioChunk chunk;
    chunk.samples.insert(chunk.samples.end(), samples.begin() + static_cast<std::ptrdiff_t>(offset),
                         samples.begin() + static_cast<std::ptrdiff_t>(offset + count));
    chunk.timestamp_ns = timestamp_ns;
    chunk.tier = "tier1";
    std::cout << "push chunk offset=" << offset << " count=" << count
              << " ts_ns=" << timestamp_ns << "\n";
    (void)pipeline.push_audio(stream, chunk);
    timestamp_ns += 100'000'000LL;
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }

  std::cout << "flushing stream\n";
  pipeline.flush_stream(stream);
  std::cout << "ending stream\n";
  pipeline.end_stream(stream);
  std::cout << "done\n";
  return 0;
}
