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

std::string load_text_file(std::string const& path) {
  std::ifstream f(path);
  if (!f)
    return {};
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

bool config_has_true_flag(std::string const& path, std::string const& key) {
  std::string const text = load_text_file(path);
  if (text.empty())
    return false;
  std::string const quoted = "\"" + key + "\"";
  auto pos = text.find(quoted);
  if (pos == std::string::npos)
    return false;
  pos = text.find(':', pos);
  if (pos == std::string::npos)
    return false;
  ++pos;
  while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos])))
    ++pos;
  return text.compare(pos, 4, "true") == 0;
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

std::string pick_existing_file(std::string const& base,
                               std::initializer_list<char const*> candidates) {
  for (char const* candidate : candidates) {
    std::string path = base + candidate;
    std::ifstream file(path, std::ios::binary);
    if (file.good()) {
      return path;
    }
  }
  return {};
}

struct WhisperRuntime {
  std::string model_dir;
  int feature_size = 80;
  std::string selected_provider;
  bool have_vocab = false;
  bool timestamped_mode = false;
  WhisperTokenIds token_ids;
  std::vector<std::string> vocab;

  Ort::Env env;
  std::unique_ptr<Ort::Session> encoder;
  std::unique_ptr<Ort::Session> decoder;
  std::unique_ptr<Ort::Session> decoder_with_past;
  std::vector<std::string> enc_in_names;
  std::vector<std::string> enc_out_names;
  std::vector<std::string> dec_in_names;
  std::vector<std::string> dec_out_names;
  std::vector<std::string> dec_past_in_names;
  std::vector<std::string> dec_present_out_names;
  std::vector<std::string> dec_encoder_present_out_names;
  std::string dec_logits_out_name;
  std::vector<std::string> dec_with_past_in_names;
  std::vector<std::string> dec_with_past_out_names;
  std::vector<std::string> dec_with_past_present_out_names;
  std::string dec_with_past_logits_out_name;
  bool encoder_first = false;
  bool decoder_uses_kv_cache = false;

  explicit WhisperRuntime(std::string model_dir_in, std::string provider_override)
      : model_dir(std::move(model_dir_in)), env(ORT_LOGGING_LEVEL_ERROR, "ears_transcribe") {
    std::string base = ensure_trailing_sep(model_dir);
    std::string enc_path = pick_existing_file(
        base, {"encoder_model.onnx", "encoder_model_fp16.onnx", "encoder_model_int8.onnx",
               "encoder_model_q4.onnx", "encoder_model_q4f16.onnx",
               "encoder_model_quantized.onnx", "encoder_model_bnb4.onnx"});
    std::string dec_path = pick_existing_file(
        base, {"decoder_model.onnx", "decoder_model_fp16.onnx", "decoder_model_int8.onnx",
               "decoder_model_q4.onnx", "decoder_model_q4f16.onnx",
               "decoder_model_quantized.onnx", "decoder_model_bnb4.onnx",
               "decoder_model_merged.onnx", "decoder_model_merged_fp16.onnx",
               "decoder_model_merged_int8.onnx", "decoder_model_merged_q4.onnx",
               "decoder_model_merged_q4f16.onnx", "decoder_model_merged_quantized.onnx",
               "decoder_model_merged_bnb4.onnx"});
    std::string dec_with_past_path = pick_existing_file(
        base, {"decoder_with_past_model.onnx", "decoder_with_past_model_fp16.onnx",
               "decoder_with_past_model_int8.onnx", "decoder_with_past_model_q4.onnx",
               "decoder_with_past_model_q4f16.onnx",
               "decoder_with_past_model_quantized.onnx", "decoder_with_past_model_bnb4.onnx"});
    std::string vocab_path = base + "vocab.txt";
    std::string preproc_path = base + "preprocessor_config.json";
    std::string generation_config_path = base + "generation_config.json";

    if (enc_path.empty() || dec_path.empty()) {
      throw std::runtime_error("Whisper ONNX model files not found in: " + model_dir);
    }

    have_vocab = load_vocab(vocab_path, vocab);
    token_ids = resolve_token_ids(vocab);
    feature_size = load_feature_size(preproc_path);
    timestamped_mode = config_has_true_flag(generation_config_path, "return_timestamps");

    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(2);
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
    std::string requested = internal::to_lower_ascii(provider_override);
    if (requested.empty()) {
      requested = internal::to_lower_ascii(internal::getenv_copy("EARS_ONNX_PROVIDER"));
    }
    if (requested.empty()) {
      requested = "auto";
    }
    selected_provider = asr_onnx::append_onnx_provider_or_cpu(opts, "onnx_runtime", requested);
    if (selected_provider.empty() || selected_provider == "cpu") {
      throw std::runtime_error("CPU execution provider is not allowed for Whisper");
    }

    std::wstring enc_w = internal::to_wide_utf8(enc_path);
    std::wstring dec_w = internal::to_wide_utf8(dec_path);
    encoder = std::make_unique<Ort::Session>(env, enc_w.c_str(), opts);
    decoder = std::make_unique<Ort::Session>(env, dec_w.c_str(), opts);
    if (!dec_with_past_path.empty()) {
      std::wstring dec_with_past_w = internal::to_wide_utf8(dec_with_past_path);
      decoder_with_past =
          std::make_unique<Ort::Session>(env, dec_with_past_w.c_str(), opts);
    }

    enc_in_names = encoder->GetConst().GetInputNames();
    enc_out_names = encoder->GetConst().GetOutputNames();
    dec_in_names = decoder->GetConst().GetInputNames();
    dec_out_names = decoder->GetConst().GetOutputNames();
    encoder_first = (dec_in_names.size() >= 2 && dec_in_names[0] != "input_ids");
    for (auto const& name : dec_in_names) {
      if (name.find("past_key_values.") == 0) {
        dec_past_in_names.push_back(name);
      }
    }
    for (auto const& name : dec_out_names) {
      if (name == "logits") {
        dec_logits_out_name = name;
      } else if (name.find("present.") == 0) {
        dec_present_out_names.push_back(name);
        if (name.find(".encoder.") != std::string::npos) {
          dec_encoder_present_out_names.push_back(name);
        }
      }
    }
    if (dec_logits_out_name.empty() && !dec_out_names.empty()) {
      dec_logits_out_name = dec_out_names.front();
    }
    decoder_uses_kv_cache = !dec_past_in_names.empty() || decoder_with_past != nullptr;
    if (decoder_with_past) {
      dec_with_past_in_names = decoder_with_past->GetConst().GetInputNames();
      dec_with_past_out_names = decoder_with_past->GetConst().GetOutputNames();
      for (auto const& name : dec_with_past_out_names) {
        if (name == "logits") {
          dec_with_past_logits_out_name = name;
        } else if (name.find("present.") == 0) {
          dec_with_past_present_out_names.push_back(name);
        }
      }
      if (dec_with_past_logits_out_name.empty() && !dec_with_past_out_names.empty()) {
        dec_with_past_logits_out_name = dec_with_past_out_names.front();
      }
    }
  }
};

struct CachedTensor {
  std::vector<float> data;
  std::vector<int64_t> shape;
};

std::string cache_output_name_for_input(std::string const& input_name) {
  constexpr char const* prefix = "past_key_values.";
  if (input_name.find(prefix) != 0) {
    return input_name;
  }
  return "present." + input_name.substr(std::char_traits<char>::length(prefix));
}

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

    bool encoder_first = runtime->encoder_first;

    // Whisper decoder prompt.
    // Multilingual: <|startoftranscript|><|en|><|transcribe|><|notimestamps|>
    // English-only: usually <|startoftranscript|><|transcribe|><|notimestamps|>
    std::vector<int64_t> input_ids;
    if (model_dir.find("tiny.en") != std::string::npos ||
        model_dir.find("whisper-tiny-en") != std::string::npos) {
      input_ids = {token_ids.start_token};
      if (!runtime->timestamped_mode) {
        input_ids.push_back(token_ids.task_transcribe);
        input_ids.push_back(token_ids.no_timestamps);
      }
    } else {
      input_ids = {token_ids.start_token};
      if (!runtime->timestamped_mode) {
        input_ids.push_back(token_ids.lang_en);
        input_ids.push_back(token_ids.task_transcribe);
        input_ids.push_back(token_ids.no_timestamps);
      }
    }
    std::string text;
    std::unordered_map<std::string, Ort::Value> kv_cache;
    std::unordered_map<std::string, CachedTensor> encoder_kv_cache;

    for (int t = 0; t < MAX_TOKENS && static_cast<int>(input_ids.size()) < MAX_DECODER_SEQ_LEN;
         ++t) {
      std::vector<int64_t> ids_shape = {1, static_cast<int64_t>(input_ids.size())};
      std::vector<int64_t> step_input_ids = input_ids;
      if (t > 0 && runtime->decoder_with_past) {
        step_input_ids.assign(1, input_ids.back());
        ids_shape[1] = 1;
      }
      std::vector<Ort::Value> dec_inputs;
      std::vector<const char*> dec_input_name_ptrs;
      std::vector<const char*> dec_output_name_ptrs;
      Ort::Session* active_decoder = runtime->decoder.get();
      std::string active_logits_name = runtime->dec_logits_out_name;

      if (t == 0 || !runtime->decoder_with_past) {
        dec_input_name_ptrs.reserve(runtime->dec_in_names.size());
        dec_output_name_ptrs.reserve(runtime->dec_out_names.size());
        for (auto const& name : runtime->dec_in_names) {
          dec_input_name_ptrs.push_back(name.c_str());
          if (name == "input_ids") {
            dec_inputs.push_back(Ort::Value::CreateTensor<int64_t>(
                mem, step_input_ids.data(), step_input_ids.size(), ids_shape.data(), ids_shape.size()));
          } else if (name == "encoder_hidden_states") {
            dec_inputs.push_back(Ort::Value::CreateTensor<float>(
                mem, const_cast<float*>(enc_ptr), enc_size, enc_out_shape.data(),
                static_cast<int>(enc_out_shape.size())));
          }
        }
        for (auto const& name : runtime->dec_out_names) {
          dec_output_name_ptrs.push_back(name.c_str());
        }
      } else {
        active_decoder = runtime->decoder_with_past.get();
        active_logits_name = runtime->dec_with_past_logits_out_name;
        dec_input_name_ptrs.reserve(runtime->dec_with_past_in_names.size());
        dec_output_name_ptrs.reserve(runtime->dec_with_past_out_names.size());
        if (dbg.is_open()) {
          dbg << "\ncache_before_t" << t << " size=" << kv_cache.size() << " ";
          for (auto const& entry : kv_cache)
            dbg << entry.first << " ";
          dbg << "\n";
        }
        for (auto const& name : runtime->dec_with_past_in_names) {
          dec_input_name_ptrs.push_back(name.c_str());
          if (name == "input_ids") {
            dec_inputs.push_back(Ort::Value::CreateTensor<int64_t>(
                mem, step_input_ids.data(), step_input_ids.size(), ids_shape.data(), ids_shape.size()));
            continue;
          }
          if (name == "encoder_hidden_states") {
            dec_inputs.push_back(Ort::Value::CreateTensor<float>(
                mem, const_cast<float*>(enc_ptr), enc_size, enc_out_shape.data(),
                static_cast<int>(enc_out_shape.size())));
            continue;
          }
          if (name.find(".encoder.") != std::string::npos) {
            auto cache_it = encoder_kv_cache.find(cache_output_name_for_input(name));
            if (cache_it == encoder_kv_cache.end()) {
              if (dbg.is_open()) {
                dbg << "missing_encoder_cache " << name
                    << " expected=" << cache_output_name_for_input(name) << "\n";
              }
              throw std::runtime_error("Missing decoder cache input: " + name);
            }
            CachedTensor& cached = cache_it->second;
            dec_inputs.push_back(Ort::Value::CreateTensor<float>(
                mem, cached.data.data(), cached.data.size(), cached.shape.data(), cached.shape.size()));
            continue;
          }
          auto cache_it = kv_cache.find(cache_output_name_for_input(name));
          if (cache_it == kv_cache.end()) {
            if (dbg.is_open()) {
              dbg << "missing_cache " << name << " expected=" << cache_output_name_for_input(name)
                  << "\n";
            }
            throw std::runtime_error("Missing decoder cache input: " + name);
          }
          dec_inputs.push_back(std::move(cache_it->second));
          kv_cache.erase(cache_it);
        }
        for (auto const& name : runtime->dec_with_past_out_names) {
          dec_output_name_ptrs.push_back(name.c_str());
        }
      }

      auto dec_out = active_decoder->Run(Ort::RunOptions{nullptr}, dec_input_name_ptrs.data(),
                                         dec_inputs.data(), dec_inputs.size(), dec_output_name_ptrs.data(),
                                         dec_output_name_ptrs.size());
      size_t logits_index = 0;
      std::vector<std::string> const& active_output_names =
          (active_decoder == runtime->decoder.get()) ? runtime->dec_out_names : runtime->dec_with_past_out_names;
      for (size_t i = 0; i < active_output_names.size(); ++i) {
        if (active_output_names[i] == active_logits_name) {
          logits_index = i;
          break;
        }
      }
      float const* logits = dec_out[logits_index].GetTensorData<float>();
      auto logits_info = dec_out[logits_index].GetTensorTypeAndShapeInfo();
      std::vector<int64_t> logits_shape = logits_info.GetShape();
      int64_t vocab_size = logits_shape.back();
      int64_t last_step = logits_shape[1] - 1;
      int64_t offset = last_step * vocab_size;
      int next_id = 0;
      float best = -1e9f;
      for (int64_t v = 0; v < vocab_size; ++v) {
        int token_id = static_cast<int>(v);
        if (token_id != token_ids.eot_token &&
            ((!runtime->timestamped_mode && is_suppressed_token(token_id, token_ids)) ||
             (runtime->timestamped_mode &&
              (token_id == token_ids.start_token || token_id == token_ids.no_timestamps))))
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
      if (have_vocab && next_id < static_cast<int>(vocab.size())) {
        tok = vocab[static_cast<size_t>(next_id)];
        if (tok.size() >= 2 && tok.compare(0, 2, "<|") == 0 && tok.find("|>") != std::string::npos)
          tok.clear();
      } else if (next_id >= 0 && next_id < 256) {
        tok = std::string(1, static_cast<char>(next_id));
      }
      if (!tok.empty())
        text += tok;
      else if (next_id >= token_ids.timestamp_begin && next_id <= token_ids.timestamp_end) {
      }
      else if (next_id >= 256 && next_id < 51865 && !text.empty() && text.back() != ' ')
        text += ' ';
      input_ids.push_back(next_id);

      std::unordered_map<std::string, Ort::Value> next_cache;
      for (size_t i = 0; i < active_output_names.size(); ++i) {
        std::string const& name = active_output_names[i];
        if (name == active_logits_name || name.find("present.") != 0) {
          continue;
        }
        next_cache.emplace(name, std::move(dec_out[i]));
      }
      if (active_decoder == runtime->decoder.get()) {
        kv_cache = std::move(next_cache);
        for (auto const& entry : kv_cache) {
          if (entry.first.find(".encoder.") == std::string::npos)
            continue;
          CachedTensor cached;
          auto info = entry.second.GetTensorTypeAndShapeInfo();
          cached.shape = info.GetShape();
          size_t const count = info.GetElementCount();
          float const* data = entry.second.GetTensorData<float>();
          cached.data.assign(data, data + count);
          encoder_kv_cache[entry.first] = std::move(cached);
        }
      } else {
        for (auto& entry : next_cache) {
          kv_cache[entry.first] = std::move(entry.second);
        }
      }
      if (dbg.is_open()) {
        dbg << "cache_after_t" << t << " size=" << kv_cache.size() << "\n";
      }
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
