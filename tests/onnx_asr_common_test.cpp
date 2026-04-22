#include "asr/onnx_asr_common.hpp"

#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <string>
#include <vector>

namespace ears {
namespace {

TEST(OnnxAsrCommonTest, LoadTokenTable_ParsesTokenIndexFormatAndBlankId) {
  std::filesystem::path dir = std::filesystem::temp_directory_path() / "ears_token_table_test";
  std::error_code ec;
  std::filesystem::remove_all(dir, ec);
  std::filesystem::create_directories(dir, ec);

  std::filesystem::path vocab = dir / "vocab.txt";
  {
    std::ofstream out(vocab);
    ASSERT_TRUE(static_cast<bool>(out));
    out << "<unk> 0\n";
    out << "the 1\n";
    out << "<blk> 2\n";
  }

  asr_onnx::TokenTable table = asr_onnx::load_token_table(dir.string());
  EXPECT_EQ(table.tokens.size(), 3u);
  EXPECT_EQ(table.tokens[0], "<unk>");
  EXPECT_EQ(table.tokens[1], "the");
  EXPECT_EQ(table.tokens[2], "<blk>");
  EXPECT_EQ(table.blank_id, 2);

  std::filesystem::remove_all(dir, ec);
}

TEST(OnnxAsrCommonTest, DecodeIdsToText_HandlesWordpieceMarkers) {
  std::vector<std::string> tokens = {
      "<blk>", std::string("\xE2\x96\x81") + "THE", "##RE", "|", "A@@", "B",
      "</w>",  std::string("\xC4\xA0") + "CAT"};
  std::vector<int64_t> ids = {1, 2, 3, 4, 5, 6, 7};
  std::string text = asr_onnx::decode_ids_to_text(ids, tokens, 0, false, true);

  EXPECT_EQ(text, "THERE AB CAT");
}

}  // namespace
}  // namespace ears
