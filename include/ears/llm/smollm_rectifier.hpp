#pragma once

#include <memory>
#include <string>

#include "ears/llm.hpp"

namespace ears {

class SmolLMRectifier : public ISemanticRectifier {
public:
  explicit SmolLMRectifier(std::string const& model_path);
  ~SmolLMRectifier() override;

  std::string rectify(std::string const& phonetic_text, std::string const& app_context,
                      std::string const& history) override;

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace ears
