#pragma once

#include "ears/llm.hpp"

namespace ears {

class DummyLlm : public ISemanticRectifier {
public:
  std::string rectify(std::string const& phonetic_text, std::string const& /*app_context*/,
                      std::string const& /*history*/) override {
    return phonetic_text;  // pass-through
  }
};

}  // namespace ears
