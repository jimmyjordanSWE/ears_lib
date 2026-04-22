#pragma once

#include <string>

namespace ears {

struct ISemanticRectifier {
  virtual ~ISemanticRectifier() = default;
  virtual std::string rectify(std::string const& phonetic_text, std::string const& app_context,
                              std::string const& history) = 0;
};

}  // namespace ears
