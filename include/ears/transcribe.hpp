#pragma once

#include <string>

namespace ears {

/**
 * Full audio-to-text transcription using Whisper ONNX (encoder + decoder).
 * Requires encoder_model.onnx and decoder_model.onnx (e.g. from Hugging Face
 * onnx-community/whisper-*).
 *
 * - model_dir: directory containing encoder_model.onnx and decoder_model.onnx
 * - samples: mono float audio, typically 16 kHz
 * - Returns transcribed text, or empty on error / when EARS_HAS_ONNX=0
 */
std::string transcribe_audio(float const* samples, size_t num_samples, int sample_rate,
                             std::string const& model_dir);
std::string transcribe_audio(float const* samples, size_t num_samples, int sample_rate,
                             std::string const& model_dir, std::string const& provider_override);

/**
 * Resolve and return the active ONNX execution provider for a model directory.
 * Honors EARS_ONNX_PROVIDER when set (e.g. cuda, dml, coreml, tensorrt, migraphx, auto).
 */
std::string transcribe_provider(std::string const& model_dir);
std::string transcribe_provider(std::string const& model_dir, std::string const& provider_override);

}  // namespace ears
