#pragma once

#include <cstdlib>
#include <string>

namespace ears::internal {

inline std::string getenv_copy(std::string const& name) {
#ifdef _WIN32
  char* value = nullptr;
  size_t value_len = 0;
  if (_dupenv_s(&value, &value_len, name.c_str()) != 0 || value == nullptr) {
    return "";
  }
  std::string out(value);
  free(value);
  return out;
#else
  char const* value = std::getenv(name.c_str());
  return value == nullptr ? std::string() : std::string(value);
#endif
}

}  // namespace ears::internal
