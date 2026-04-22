#pragma once

#include <limits>
#include <string>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#endif

namespace ears::internal {

inline std::wstring to_wide_utf8(std::string const& text) {
#ifdef _WIN32
  if (text.empty()) {
    return std::wstring();
  }
  if (text.size() > static_cast<size_t>((std::numeric_limits<int>::max)())) {
    return std::wstring();
  }

  int const input_len = static_cast<int>(text.size());
  int required =
      MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, text.data(), input_len, nullptr, 0);
  if (required <= 0) {
    required = MultiByteToWideChar(CP_UTF8, 0, text.data(), input_len, nullptr, 0);
    if (required <= 0) {
      return std::wstring();
    }
  }

  std::wstring out(static_cast<size_t>(required), L'\0');
  int converted = MultiByteToWideChar(CP_UTF8, 0, text.data(), input_len, out.data(), required);
  if (converted <= 0) {
    return std::wstring();
  }
  return out;
#else
  std::wstring out;
  out.reserve(text.size());
  for (unsigned char c : text) {
    out.push_back(static_cast<wchar_t>(c));
  }
  return out;
#endif
}

}  // namespace ears::internal
