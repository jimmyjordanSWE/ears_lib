#pragma once

#include <algorithm>
#include <cctype>
#include <string>

namespace ears::internal {

inline std::string to_lower_ascii(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value;
}

inline std::string trim_copy(std::string const& input) {
  size_t begin = 0;
  while (begin < input.size() && std::isspace(static_cast<unsigned char>(input[begin])) != 0) {
    ++begin;
  }
  size_t end = input.size();
  while (end > begin && std::isspace(static_cast<unsigned char>(input[end - 1])) != 0) {
    --end;
  }
  return input.substr(begin, end - begin);
}

inline bool parse_bool_strict(std::string const& value, bool& out) {
  std::string lowered = to_lower_ascii(trim_copy(value));
  if (lowered == "1" || lowered == "true" || lowered == "yes" || lowered == "on") {
    out = true;
    return true;
  }
  if (lowered == "0" || lowered == "false" || lowered == "no" || lowered == "off") {
    out = false;
    return true;
  }
  return false;
}

inline bool parse_int_strict(std::string const& value, int& out) {
  try {
    std::string trimmed = trim_copy(value);
    size_t parsed = 0;
    int const converted = std::stoi(trimmed, &parsed);
    if (parsed != trimmed.size()) {
      return false;
    }
    out = converted;
    return true;
  } catch (...) {
    return false;
  }
}

inline bool parse_float_strict(std::string const& value, float& out) {
  try {
    std::string trimmed = trim_copy(value);
    size_t parsed = 0;
    float const converted = std::stof(trimmed, &parsed);
    if (parsed != trimmed.size()) {
      return false;
    }
    out = converted;
    return true;
  } catch (...) {
    return false;
  }
}

}  // namespace ears::internal
