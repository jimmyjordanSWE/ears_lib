#include "ears/types.hpp"

#include <gtest/gtest.h>

namespace ears {
namespace {

TEST(TypesTest, AudioChunk_DefaultConstruct_HasExpectedDefaults) {
  AudioChunk chunk;
  EXPECT_TRUE(chunk.samples.empty());
  EXPECT_EQ(chunk.timestamp_ns, 0);
  EXPECT_EQ(chunk.tier, "tier1");
}

TEST(TypesTest, AudioChunk_CanHoldSamples) {
  AudioChunk chunk;
  chunk.samples = {0.1f, 0.2f, 0.3f};
  chunk.timestamp_ns = 12345;
  chunk.tier = "tier2";

  ASSERT_EQ(chunk.samples.size(), 3u);
  EXPECT_FLOAT_EQ(chunk.samples[0], 0.1f);
  EXPECT_FLOAT_EQ(chunk.samples[1], 0.2f);
  EXPECT_FLOAT_EQ(chunk.samples[2], 0.3f);
  EXPECT_EQ(chunk.timestamp_ns, 12345);
  EXPECT_EQ(chunk.tier, "tier2");
}

TEST(TypesTest, Context_DefaultConstruct_EmptyStrings) {
  Context ctx;
  EXPECT_TRUE(ctx.app_window_title.empty());
  EXPECT_TRUE(ctx.process_name.empty());
}

TEST(TypesTest, Context_CanSetAppWindowTitle) {
  Context ctx;
  ctx.app_window_title = "Visual Studio Code";
  ctx.process_name = "code.exe";

  EXPECT_EQ(ctx.app_window_title, "Visual Studio Code");
  EXPECT_EQ(ctx.process_name, "code.exe");
}

TEST(TypesTest, TranscriptionResult_DefaultConstruct_HasExpectedDefaults) {
  TranscriptionResult result;
  EXPECT_TRUE(result.corrected_text.empty());
  EXPECT_TRUE(result.raw_asr_text.empty());
  EXPECT_TRUE(result.raw_asr_json.empty());
  EXPECT_EQ(result.timestamp_ns, 0);
  EXPECT_FALSE(result.is_final);
}

TEST(TypesTest, TranscriptionResult_CanSetAllFields) {
  TranscriptionResult result;
  result.corrected_text = "corrected";
  result.raw_asr_text = "raw";
  result.raw_asr_json = "{}";
  result.timestamp_ns = 999;
  result.is_final = true;

  EXPECT_EQ(result.corrected_text, "corrected");
  EXPECT_EQ(result.raw_asr_text, "raw");
  EXPECT_EQ(result.raw_asr_json, "{}");
  EXPECT_EQ(result.timestamp_ns, 999);
  EXPECT_TRUE(result.is_final);
}

}  // namespace
}  // namespace ears
