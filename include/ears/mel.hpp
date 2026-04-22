#pragma once

#include <cstddef>
#include <vector>

namespace ears {

/**
 * Compute log-mel spectrogram for Whisper-style ASR.
 * - Sample rate: 16000 Hz
 * - Window: 25 ms (400 samples), hop: 10 ms (160 samples)
 * - FFT: 512 points (400 samples zero-padded)
 * - Mel: 80 bins, 0–8000 Hz
 *
 * Returns shape (80, num_frames), row-major, float.
 * num_frames = (num_samples - 400) / 160 + 1 (clamped to >= 1).
 */
std::vector<float> compute_log_mel_80(float const* samples, size_t num_samples);

/**
 * Compute log-mel spectrogram with configurable mel bin count.
 * Returns shape (num_mels, num_frames), row-major, float.
 */
std::vector<float> compute_log_mel(float const* samples, size_t num_samples, int num_mels);

}  // namespace ears
