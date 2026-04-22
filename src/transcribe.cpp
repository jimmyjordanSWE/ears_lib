#if EARS_HAS_ONNX

#include "ears/transcribe.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <onnxruntime_cxx_api.h>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "asr/onnx_asr_common.hpp"
#include "ears/internal/env_utils.hpp"
#include "ears/internal/string_utils.hpp"
#include "ears/internal/unicode_utils.hpp"
#include "ears/mel.hpp"

namespace ears {

namespace {

constexpr int MAX_FRAMES = 3000;  // 30 s at 10 ms hop
constexpr int MAX_TOKENS = 1024;
constexpr int MAX_DECODER_SEQ_LEN = 448;

struct WhisperTokenIds {
  int start_token = 50258;
  int eot_token = 50257;
  int lang_en = 50259;
  int task_transcribe = 50359;
  int no_timestamps = 50363;
  int timestamp_begin = 50364;
  int timestamp_end = 51864;
};

bool load_vocab(std::string const& path, std::vector<std::string>& out) {
  std::ifstream f(path);
  if (!f)
    return false;
  out.clear();
  std::string line;
  while (std::getline(f, line)) {
    while (!line.empty() && (line.back() == '\r' || line.back() == '\n'))
      line.pop_back();
    out.push_back(line);
  }
  return out.size() > 256;
}

int load_feature_size(std::string const& preprocessor_config_path) {
  std::ifstream f(preprocessor_config_path);
  if (!f)
    return 80;
  std::stringstream ss;
  ss << f.rdbuf();
  std::string text = ss.str();
  std::string key = "\"feature_size\"";
  auto pos = text.find(key);
  if (pos == std::string::npos)
    return 80;
  pos = text.find(':', pos);
  if (pos == std::string::npos)
    return 80;
  ++pos;
  while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos])))
    ++pos;
  size_t end = pos;
  while (end < text.size() && std::isdigit(static_cast<unsigned char>(text[end])))
    ++end;
  if (end <= pos)
    return 80;
  try {
    int value = std::stoi(text.substr(pos, end - pos));
    if (value > 0)
      return value;
  } catch (...) {
    return 80;
  }
  return 80;
}

int find_token_id(std::vector<std::string> const& vocab, std::string const& token) {
  auto it = std::find(vocab.begin(), vocab.end(), token);
  if (it == vocab.end())
    return -1;
  return static_cast<int>(std::distance(vocab.begin(), it));
}

WhisperTokenIds resolve_token_ids(std::vector<std::string> const& vocab) {
  WhisperTokenIds ids;
  if (vocab.empty())
    return ids;

  int found = find_token_id(vocab, "<|startoftranscript|>");
  if (found >= 0)
    ids.start_token = found;
  found = find_token_id(vocab, "<|endoftext|>");
  if (found >= 0)
    ids.eot_token = found;
  found = find_token_id(vocab, "<|en|>");
  if (found >= 0)
    ids.lang_en = found;
  found = find_token_id(vocab, "<|transcribe|>");
  if (found >= 0)
    ids.task_transcribe = found;
  found = find_token_id(vocab, "<|notimestamps|>");
  if (found >= 0)
    ids.no_timestamps = found;
  found = find_token_id(vocab, "<|0.00|>");
  if (found >= 0)
    ids.timestamp_begin = found;
  found = find_token_id(vocab, "<|30.00|>");
  if (found >= 0) {
    ids.timestamp_end = found;
  } else {
    ids.timestamp_end = ids.timestamp_begin + 1500;
  }

  return ids;
}

bool is_suppressed_token(int token_id, WhisperTokenIds const& ids) {
  if (token_id == ids.start_token || token_id == ids.lang_en || token_id == ids.task_transcribe ||
      token_id == ids.no_timestamps) {
    return true;
  }
  return token_id >= ids.timestamp_begin && token_id <= ids.timestamp_end;
}

std::string ensure_trailing_sep(std::string path) {
  if (!path.empty() && path.back() != '/' && path.back() != '\\')
    path += "/";
  return path;
}

bool env_truthy(char const* name) {
  std::string raw = internal::getenv_copy(name);
  if (raw.empty())
    return false;
  std::string value = internal::to_lower_ascii(raw);
  return value == "1" || value == "true" || value == "yes" || value == "on";
}

struct WhisperRuntime {
  std::string model_dir;
  int feature_size = 80;
  std::string selected_provider;
  bool have_vocab = false;
  WhisperTokenIds token_ids;
  std::vector<std::string> vocab;

  Ort::Env env;
  std::unique_ptr<Ort::Session> encoder;
  std::unique_ptr<Ort::Session> decoder;
  std::vector<std::string> enc_in_names;
  std::vector<std::string> enc_out_names;
  std::vector<std::string> dec_in_names;
  std::string dec_out_name;
  bool encoder_first = false;

  explicit WhisperRuntime(std::string model_dir_in, std::string provider_override)
      : model_dir(std::move(model_dir_in)), env(ORT_LOGGING_LEVEL_WARNING, "ears_transcribe") {
    std::string base = ensure_trailing_sep(model_dir);
    std::string enc_path = base + "encoder_model.onnx";
    std::string dec_path = base + "decoder_model.onnx";
    std::string vocab_path = base + "vocab.txt";
    std::string preproc_path = base + "preprocessor_config.json";

    have_vocab = load_vocab(vocab_path, vocab);
    token_ids = resolve_token_ids(vocab);
    feature_size = load_feature_size(preproc_path);

    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(2);
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    bool allow_cpu_ep_fallback = env_truthy("EARS_ALLOW_CPU_EP_FALLBACK");
    opts.AddConfigEntry("session.disable_cpu_ep_fallback", allow_cpu_ep_fallback ? "0" : "1");
    std::string requested = internal::to_lower_ascii(provider_override);
    if (requested.empty()) {
      requested = internal::to_lower_ascii(internal::getenv_copy("EARS_ONNX_PROVIDER"));
    }
    if (requested.empty()) {
      requested = "auto";
    }
    selected_provider = asr_onnx::append_onnx_provider_or_cpu(opts, "onnx_runtime", requested);

    std::wstring enc_w = internal::to_wide_utf8(enc_path);
    std::wstring dec_w = internal::to_wide_utf8(dec_path);
    encoder = std::make_unique<Ort::Session>(env, enc_w.c_str(), opts);
    decoder = std::make_unique<Ort::Session>(env, dec_w.c_str(), opts);

    enc_in_names = encoder->GetConst().GetInputNames();
    enc_out_names = encoder->GetConst().GetOutputNames();
    dec_in_names = decoder->GetConst().GetInputNames();
    dec_out_name = decoder->GetConst().GetOutputNames()[0];
    encoder_first = (dec_in_names.size() >= 2 && dec_in_names[0] != "input_ids");
  }
};

std::shared_ptr<WhisperRuntime> get_runtime(std::string const& model_dir,
                                            std::string const& provider_override) {
  static std::mutex runtime_mu;
  static std::unordered_map<std::string, std::shared_ptr<WhisperRuntime>> runtimes;
  std::string const key = model_dir + "::" + asr_onnx::normalize_provider_id(provider_override);

  std::lock_guard<std::mutex> lock(runtime_mu);
  auto it = runtimes.find(key);
  if (it != runtimes.end())
    return it->second;

  auto runtime = std::make_shared<WhisperRuntime>(model_dir, provider_override);
  runtimes.emplace(key, runtime);
  return runtime;
}

}  // namespace

std::string transcribe_audio(float const* samples, size_t num_samples, int sample_rate,
                             std::string const& model_dir) {
  return transcribe_audio(samples, num_samples, sample_rate, model_dir, "");
}

std::string transcribe_audio(float const* samples, size_t num_samples, int sample_rate,
                             std::string const& model_dir, std::string const& provider_override) {
  if (samples == nullptr || num_samples == 0 || sample_rate != 16000)
    return "";
  if (model_dir.empty())
    return "";

  try {
    auto runtime = get_runtime(model_dir, provider_override);
    WhisperTokenIds const& token_ids = runtime->token_ids;
    bool have_vocab = runtime->have_vocab;
    std::vector<std::string> const& vocab = runtime->vocab;

    int feature_size = std::max(1, runtime->feature_size);
    std::vector<float> mel = compute_log_mel(samples, num_samples, feature_size);
    int num_frames = static_cast<int>(mel.size()) / feature_size;
    if (num_frames <= 0)
      return "";

    int use_frames = std::min(num_frames, MAX_FRAMES);
    size_t valid = static_cast<size_t>(feature_size * use_frames);
    float mel_max = -1e9f;
    for (size_t i = 0; i < valid; ++i) {
      if (mel[i] > mel_max)
        mel_max = mel[i];
    }
    if (mel_max < -1e8f)
      mel_max = 0.f;
    float clip_min = mel_max - 8.f;
    for (size_t i = 0; i < valid; ++i) {
      float& v = mel[i];
      if (v < clip_min)
        v = clip_min;
      v = (v + 4.f) / 4.f;  // Whisper formula
    }

    float pad_scaled = (clip_min + 4.f) / 4.f;
    std::vector<float> mel_input(static_cast<size_t>(feature_size * MAX_FRAMES), pad_scaled);
    for (int f = 0; f < use_frames; ++f) {
      for (int m = 0; m < feature_size; ++m) {
        mel_input[static_cast<size_t>(m * MAX_FRAMES + f)] =
            mel[static_cast<size_t>(f * feature_size + m)];
      }
    }

    std::vector<int64_t> enc_shape = {1, feature_size, MAX_FRAMES};
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value enc_input = Ort::Value::CreateTensor<float>(mem, mel_input.data(), mel_input.size(),
                                                           enc_shape.data(), enc_shape.size());

    std::vector<const char*> enc_in_name_ptrs;
    std::vector<const char*> enc_out_name_ptrs;
    for (auto const& n : runtime->enc_in_names)
      enc_in_name_ptrs.push_back(n.c_str());
    for (auto const& n : runtime->enc_out_names)
      enc_out_name_ptrs.push_back(n.c_str());
    auto enc_out = runtime->encoder->Run(Ort::RunOptions{nullptr}, enc_in_name_ptrs.data(),
                                         &enc_input, enc_in_name_ptrs.size(),
                                         enc_out_name_ptrs.data(), enc_out_name_ptrs.size());
    Ort::Value& enc_hidden = enc_out[0];
    auto enc_info = enc_hidden.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> enc_out_shape = enc_info.GetShape();
    size_t enc_size = static_cast<size_t>(enc_info.GetElementCount());
    float const* enc_ptr = enc_hidden.GetTensorData<float>();

    std::ofstream dbg;
    if (!internal::getenv_copy("EARS_TRANSCRIBE_DEBUG").empty()) {
      dbg.open("transcribe_debug.txt");
      if (dbg) {
        dbg << "enc_shape ";
        for (auto d : enc_out_shape)
          dbg << d << " ";
        dbg << "\n";
      }
    }

    std::vector<const char*> dec_in_name_ptrs;
    for (auto const& n : runtime->dec_in_names)
      dec_in_name_ptrs.push_back(n.c_str());
    const char* dec_out_name_ptr = runtime->dec_out_name.c_str();
    bool encoder_first = runtime->encoder_first;

    // Whisper decoder prompt.
    // Multilingual: <|startoftranscript|><|en|><|transcribe|><|notimestamps|>
    // English-only: usually <|startoftranscript|><|transcribe|><|notimestamps|>
    std::vector<int64_t> input_ids;
    if (model_dir.find("tiny.en") != std::string::npos ||
        model_dir.find("whisper-tiny-en") != std::string::npos)
      input_ids = {token_ids.start_token, token_ids.task_transcribe, token_ids.no_timestamps};
    else
      input_ids = {token_ids.start_token, token_ids.lang_en, token_ids.task_transcribe,
                   token_ids.no_timestamps};
    std::string text;

    for (int t = 0; t < MAX_TOKENS && static_cast<int>(input_ids.size()) < MAX_DECODER_SEQ_LEN;
         ++t) {
      std::vector<int64_t> ids_shape = {1, static_cast<int64_t>(input_ids.size())};
      Ort::Value dec_ids = Ort::Value::CreateTensor<int64_t>(
          mem, input_ids.data(), input_ids.size(), ids_shape.data(), ids_shape.size());

      std::vector<Ort::Value> dec_inputs;
      dec_inputs.push_back(Ort::Value::CreateTensor<int64_t>(
          mem, input_ids.data(), input_ids.size(), ids_shape.data(), ids_shape.size()));
      dec_inputs.push_back(Ort::Value::CreateTensor<float>(mem, const_cast<float*>(enc_ptr),
                                                           enc_size, enc_out_shape.data(),
                                                           static_cast<int>(enc_out_shape.size())));

      std::vector<Ort::Value> dec_in_vals;
      if (encoder_first) {
        dec_in_vals.push_back(std::move(dec_inputs[1]));
        dec_in_vals.push_back(std::move(dec_inputs[0]));
      } else {
        dec_in_vals.push_back(std::move(dec_inputs[0]));
        dec_in_vals.push_back(std::move(dec_inputs[1]));
      }

      auto dec_out = runtime->decoder->Run(Ort::RunOptions{nullptr}, dec_in_name_ptrs.data(),
                                           dec_in_vals.data(), 2, &dec_out_name_ptr, 1);
      float const* logits = dec_out[0].GetTensorData<float>();
      auto logits_info = dec_out[0].GetTensorTypeAndShapeInfo();
      std::vector<int64_t> logits_shape = logits_info.GetShape();
      int64_t vocab_size = logits_shape.back();
      int64_t last_step = logits_shape[1] - 1;
      int64_t offset = last_step * vocab_size;
      int next_id = 0;
      float best = -1e9f;
      for (int64_t v = 0; v < vocab_size; ++v) {
        int token_id = static_cast<int>(v);
        if (token_id != token_ids.eot_token && is_suppressed_token(token_id, token_ids))
          continue;
        float l = logits[offset + v];
        if (l > best) {
          best = l;
          next_id = token_id;
        }
      }

      if (next_id == token_ids.eot_token)
        break;
      if (t < 25 && dbg.is_open())
        dbg << "t" << t << "=" << next_id << " ";
      std::string tok;
      if (next_id >= 0 && next_id < 256)
        tok = std::string(1, static_cast<char>(next_id));
      else if (have_vocab && next_id < static_cast<int>(vocab.size())) {
        tok = vocab[static_cast<size_t>(next_id)];
        if (tok.size() >= 2 && tok.compare(0, 2, "<|") == 0 && tok.find("|>") != std::string::npos)
          tok.clear();
      }
      if (!tok.empty())
        text += tok;
      else if (next_id >= 256 && next_id < 51865 && !text.empty() && text.back() != ' ')
        text += ' ';
      input_ids.push_back(next_id);
    }

    // Normalize BPE space (Ġ U+0120 = UTF-8 C4 A0) to ASCII space
    std::string out;
    out.reserve(text.size());
    for (size_t i = 0; i < text.size(); ++i) {
      if (text[i] == '\xc4' && i + 1 < text.size() && text[i + 1] == '\xa0') {
        out += ' ';
        ++i;
      } else
        out += text[i];
    }
    while (!out.empty() && (out.front() == ' ' || out.front() == '\t'))
      out.erase(0, 1);
    while (!out.empty() && (out.back() == ' ' || out.back() == '\t'))
      out.pop_back();
    if (dbg.is_open())
      dbg << "\ntext [" << out << "]\n";
    return out;
  } catch (Ort::Exception const& e) {
    throw std::runtime_error(std::string("ONNX: ") + e.what());
  }
}

std::string transcribe_provider(std::string const& model_dir) {
  return transcribe_provider(model_dir, "");
}

std::string transcribe_provider(std::string const& model_dir,
                                std::string const& provider_override) {
  if (model_dir.empty())
    return "";
  auto runtime = get_runtime(model_dir, provider_override);
  return runtime ? runtime->selected_provider : "";
}

}  // namespace ears

#else  // !EARS_HAS_ONNX

#include "ears/transcribe.hpp"

namespace ears {

std::string transcribe_audio(float const* /*samples*/, size_t /*num_samples*/, int /*sample_rate*/,
                             std::string const& /*model_dir*/) {
  return "";
}

std::string transcribe_audio(float const* /*samples*/, size_t /*num_samples*/, int /*sample_rate*/,
                             std::string const& /*model_dir*/,
                             std::string const& /*provider_override*/) {
  return "";
}

std::string transcribe_provider(std::string const& /*model_dir*/) {
  return "";
}

std::string transcribe_provider(std::string const& /*model_dir*/,
                                std::string const& /*provider_override*/) {
  return "";
}

}  // namespace ears

#endif  // EARS_HAS_ONNX
