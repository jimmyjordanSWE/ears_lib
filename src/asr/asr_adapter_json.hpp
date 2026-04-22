#pragma once

#include <nlohmann/json.hpp>
#include <string>
#include <utility>

namespace ears::asr_json {

inline std::string build_delegate_metadata(std::string const& adapter,
                                           std::string const& relation_key,
                                           std::string const& relation_value,
                                           std::string const& provider_requested,
                                           std::string const& nested_key,
                                           std::string const& nested_json) {
  nlohmann::json meta = {
      {"adapter", adapter},
      {relation_key, relation_value},
      {"provider_requested", provider_requested},
  };

  if (!nested_json.empty()) {
    nlohmann::json parsed = nlohmann::json::parse(nested_json, nullptr, false);
    if (!parsed.is_discarded()) {
      meta[nested_key] = std::move(parsed);
    }
  }

  return meta.dump();
}

inline std::string build_status_metadata(std::string const& adapter, std::string const& runtime,
                                         std::string const& provider_requested,
                                         std::string const& decode) {
  nlohmann::json meta = {
      {"adapter", adapter},
      {"runtime", runtime},
      {"provider_requested", provider_requested},
      {"decode", decode},
  };
  return meta.dump();
}

}  // namespace ears::asr_json
