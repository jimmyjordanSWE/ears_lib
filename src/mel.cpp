#include "ears/mel.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace ears {

namespace {

constexpr int N_FFT = 400;
constexpr int WIN_SAMPLES = 400;
constexpr int HOP_SAMPLES = 160;
constexpr int N_FFT_BINS = N_FFT / 2;  // 200 - Whisper uses [..., :-1], drops last bin
constexpr int SAMPLE_RATE = 16000;
constexpr double FMAX = 8000.0;
constexpr double FMIN = 0.0;
constexpr double PI = 3.14159265358979323846;

inline double hz_to_mel(double hz) {
  return 2595.0 * std::log10(1.0 + hz / 700.0);
}
inline double mel_to_hz(double mel) {
  return 700.0 * (std::pow(10.0, mel / 2595.0) - 1.0);
}

// 400-point DFT for bins 0..N_FFT_BINS-1 (real input → magnitude²).
void dft_400_mag2(double const* x, int n_bins, double* out) {
  for (int k = 0; k < n_bins; ++k) {
    double re = 0.0, im = 0.0;
    for (int n = 0; n < N_FFT; ++n) {
      double angle = -2.0 * PI * k * n / N_FFT;
      re += x[n] * std::cos(angle);
      im += x[n] * std::sin(angle);
    }
    out[k] = (re * re + im * im);
  }
}

// Build mel filterbank: (N_MELS, N_FFT_BINS). Row i = filter for mel band i.
std::vector<std::vector<float>> build_mel_filters(int n_mels) {
  double low = hz_to_mel(FMIN);
  double high = hz_to_mel(FMAX);
  int n_fft_bins = N_FFT_BINS;
  double bin_hz = static_cast<double>(SAMPLE_RATE) / N_FFT;

  std::vector<double> mel_points(n_mels + 2);
  for (int i = 0; i < n_mels + 2; ++i)
    mel_points[i] = mel_to_hz(low + (high - low) * i / (n_mels + 1));

  std::vector<int> bin_indices(n_mels + 2);
  for (int i = 0; i < n_mels + 2; ++i)
    bin_indices[i] = static_cast<int>(std::round(mel_points[i] / bin_hz));
  bin_indices[0] = std::max(0, bin_indices[0]);
  bin_indices[n_mels + 1] = std::min(n_fft_bins - 1, bin_indices[n_mels + 1]);

  std::vector<std::vector<float>> filters(n_mels, std::vector<float>(n_fft_bins, 0.f));
  for (int i = 0; i < n_mels; ++i) {
    int left = bin_indices[i];
    int center = bin_indices[i + 1];
    int right = bin_indices[i + 2];
    for (int j = left; j < center; ++j)
      filters[i][j] = static_cast<float>(static_cast<double>(j - left) / (center - left));
    for (int j = center; j < right; ++j)
      filters[i][j] = static_cast<float>(static_cast<double>(right - j) / (right - center));
  }
  return filters;
}

bool load_whisper_mel_filters_80(std::vector<std::vector<float>>& out_filters) {
  constexpr int N_MELS_80 = 80;
  std::vector<std::string> candidates = {
      "config/whisper_mel_80x201.txt",
      "../config/whisper_mel_80x201.txt",
      "../../config/whisper_mel_80x201.txt",
  };
  std::ifstream f;
  for (auto const& p : candidates) {
    f.open(p);
    if (f.is_open())
      break;
    f.clear();
  }
  if (!f.is_open())
    return false;

  // File format: 201 rows x 80 columns (freq-major from transformers);
  // convert to 80 x 200 (mel-major, drop Nyquist bin) to match magnitudes[..., :-1].
  std::vector<std::vector<float>> freq_major;
  std::string line;
  while (std::getline(f, line)) {
    if (line.empty())
      continue;
    std::istringstream iss(line);
    std::vector<float> row;
    float v = 0.f;
    while (iss >> v)
      row.push_back(v);
    if (!row.empty())
      freq_major.push_back(std::move(row));
  }
  if (freq_major.size() < 200)
    return false;
  for (size_t i = 0; i < 200; ++i) {
    if (freq_major[i].size() < static_cast<size_t>(N_MELS_80))
      return false;
  }

  out_filters.assign(N_MELS_80, std::vector<float>(N_FFT_BINS, 0.f));
  for (int m = 0; m < N_MELS_80; ++m) {
    for (int k = 0; k < N_FFT_BINS; ++k) {
      out_filters[m][k] = freq_major[static_cast<size_t>(k)][static_cast<size_t>(m)];
    }
  }
  return true;
}

}  // namespace

std::vector<float> compute_log_mel(float const* samples, size_t num_samples, int num_mels) {
  if (num_mels <= 0)
    return {};

  static std::vector<std::vector<float>> mel_filters_80 = [] {
    std::vector<std::vector<float>> f;
    if (load_whisper_mel_filters_80(f))
      return f;
    return build_mel_filters(80);
  }();
  static std::vector<std::vector<float>> mel_filters_128 = build_mel_filters(128);

  std::vector<std::vector<float>> const* mel_filters = nullptr;
  if (num_mels == 80)
    mel_filters = &mel_filters_80;
  else if (num_mels == 128)
    mel_filters = &mel_filters_128;
  else {
    static std::vector<std::vector<float>> dynamic_filters;
    dynamic_filters = build_mel_filters(num_mels);
    mel_filters = &dynamic_filters;
  }

  int n_fft_bins = N_FFT_BINS;

  // Match Whisper-style centered STFT framing (torch.stft center=True):
  // frames = floor(num_samples / hop) + 1
  int num_frames = static_cast<int>(num_samples / HOP_SAMPLES) + 1;
  num_frames = std::max(1, num_frames);

  std::vector<float> out(static_cast<size_t>(num_mels * num_frames), 0.f);

  std::vector<double> window(WIN_SAMPLES);
  for (int i = 0; i < WIN_SAMPLES; ++i)
    window[i] = 0.5 * (1.0 - std::cos(2.0 * PI * i / (WIN_SAMPLES - 1)));

  std::vector<double> framed(N_FFT, 0.0);
  std::vector<double> power(static_cast<size_t>(n_fft_bins));

  auto reflect_index = [num_samples](int idx) -> int {
    if (num_samples == 0)
      return 0;
    int n = static_cast<int>(num_samples);
    while (idx < 0 || idx >= n) {
      if (idx < 0)
        idx = -idx;
      else
        idx = (2 * n - 2) - idx;
    }
    return idx;
  };

  for (int f = 0; f < num_frames; ++f) {
    int start = f * HOP_SAMPLES - (N_FFT / 2);
    for (int i = 0; i < N_FFT; ++i) {
      int src = reflect_index(start + i);
      framed[i] = static_cast<double>(samples[src]) * window[i];
    }

    dft_400_mag2(framed.data(), n_fft_bins, power.data());

    for (int m = 0; m < num_mels; ++m) {
      double sum = 0.0;
      for (int k = 0; k < n_fft_bins; ++k)
        sum += (*mel_filters)[m][k] * power[k];
      float log_val = static_cast<float>(std::log10(std::max(sum, 1e-10)));
      out[static_cast<size_t>(f * num_mels + m)] = log_val;
    }
  }

  return out;
}

std::vector<float> compute_log_mel_80(float const* samples, size_t num_samples) {
  return compute_log_mel(samples, num_samples, 80);
}

}  // namespace ears
