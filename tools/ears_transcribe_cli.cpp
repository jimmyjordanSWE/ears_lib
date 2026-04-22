#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "ears/transcribe.hpp"

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

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "usage: ears_transcribe_cli <model_dir> <audio_f32le> [provider]\n";
    return 1;
  }

  std::string const model_dir = argv[1];
  std::string const audio_path = argv[2];
  std::string const provider = argc >= 4 ? argv[3] : "auto";

  std::vector<float> samples;
  if (!read_f32le_file(audio_path, samples)) {
    std::cerr << "failed to read audio file: " << audio_path << "\n";
    return 2;
  }

  try {
    std::string const text = ears::transcribe_audio(samples.data(), samples.size(), 16000,
                                                    model_dir, provider);
    std::cout << text << "\n";
    return text.empty() ? 3 : 0;
  } catch (std::exception const& e) {
    std::cerr << "transcription failed: " << e.what() << "\n";
    return 4;
  }
}
