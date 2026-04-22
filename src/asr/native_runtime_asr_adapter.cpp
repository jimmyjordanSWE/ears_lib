#include "ears/asr/native_runtime_asr_adapter.hpp"

#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <sstream>
#include <string>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#include "ears/internal/env_utils.hpp"
#include "ears/internal/runtime_id_utils.hpp"
#include "ears/internal/string_utils.hpp"

#ifdef _WIN32
#include <Windows.h>
#include <shellapi.h>
#else
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace ears {

namespace {

constexpr int kRunnerContractVersion = 1;
constexpr int kDefaultRunnerTimeoutMs = 10000;
constexpr int kDefaultProbeTimeoutMs = 2500;
constexpr size_t kDefaultRunnerOutputTailBytes = 4096;

class TailBuffer {
public:
  explicit TailBuffer(size_t max_bytes) : max_bytes_(max_bytes) {}

  void append(char const* data, size_t size) {
    if (data == nullptr || size == 0) {
      return;
    }
    total_bytes_ += size;
    if (max_bytes_ == 0) {
      truncated_ = true;
      return;
    }
    if (size >= max_bytes_) {
      data_.assign(data + (size - max_bytes_), max_bytes_);
      truncated_ = true;
      return;
    }
    if (data_.size() + size > max_bytes_) {
      size_t const drop = data_.size() + size - max_bytes_;
      data_.erase(0, drop);
      truncated_ = true;
    }
    data_.append(data, size);
  }

  std::string const& data() const {
    return data_;
  }
  size_t total_bytes() const {
    return total_bytes_;
  }
  bool truncated() const {
    return truncated_ || total_bytes_ > data_.size();
  }

private:
  size_t max_bytes_ = 0;
  size_t total_bytes_ = 0;
  bool truncated_ = false;
  std::string data_;
};

struct RunnerCommand {
  std::string raw;
  std::vector<std::string> argv;
  std::string error;
};

struct ProcessResult {
  bool launched = false;
  bool timed_out = false;
  bool killed = false;
  int exit_code = 0;
  std::string error;
  std::string stdout_tail;
  std::string stderr_tail;
  size_t stdout_bytes = 0;
  size_t stderr_bytes = 0;
  bool stdout_truncated = false;
  bool stderr_truncated = false;
};

std::string trim_env(std::string const& name) {
  return internal::trim_copy(internal::getenv_copy(name));
}

bool parse_positive_int_env(std::string const& name, int min_value, int max_value, int* out_value) {
  std::string const raw = trim_env(name);
  if (raw.empty()) {
    return false;
  }
  int parsed = 0;
  if (!internal::parse_int_strict(raw, parsed)) {
    return false;
  }
  if (parsed < min_value || parsed > max_value) {
    return false;
  }
  if (out_value != nullptr) {
    *out_value = parsed;
  }
  return true;
}

size_t parse_output_tail_limit_bytes() {
  int configured = static_cast<int>(kDefaultRunnerOutputTailBytes);
  if (!parse_positive_int_env("EARS_NATIVE_RUNNER_OUTPUT_MAX_BYTES", 256, 131072, &configured)) {
    configured = static_cast<int>(kDefaultRunnerOutputTailBytes);
  }
  return static_cast<size_t>(configured);
}

std::string runner_timeout_env_name(std::string const& runtime_id) {
  std::string const normalized = internal::normalize_runtime_id(runtime_id);
  if (normalized == "tensorrt") {
    return "EARS_TENSORRT_ASR_RUNNER_TIMEOUT_MS";
  }
  if (normalized == "openvino") {
    return "EARS_OPENVINO_ASR_RUNNER_TIMEOUT_MS";
  }
  if (normalized == "coreml") {
    return "EARS_COREML_ASR_RUNNER_TIMEOUT_MS";
  }
  if (normalized == "qnn") {
    return "EARS_QNN_ASR_RUNNER_TIMEOUT_MS";
  }
  return "EARS_NATIVE_ASR_RUNNER_TIMEOUT_MS";
}

int resolve_runner_timeout_ms(std::string const& runtime_id, bool probe_mode) {
  int timeout_ms = probe_mode ? kDefaultProbeTimeoutMs : kDefaultRunnerTimeoutMs;
  int runtime_override = 0;
  if (parse_positive_int_env(runner_timeout_env_name(runtime_id), 50, 300000, &runtime_override)) {
    timeout_ms = runtime_override;
  } else {
    int global_override = 0;
    if (parse_positive_int_env("EARS_NATIVE_RUNNER_TIMEOUT_MS", 50, 300000, &global_override)) {
      timeout_ms = global_override;
    }
  }
  return timeout_ms;
}

std::atomic<uint64_t>& temp_counter() {
  static std::atomic<uint64_t> counter{1};
  return counter;
}

std::filesystem::path make_temp_base_path() {
  std::filesystem::path root = std::filesystem::temp_directory_path();
  uint64_t const id = temp_counter().fetch_add(1);
  std::ostringstream name;
  name << "ears_native_asr_" << id;
  return root / name.str();
}

class ScopedTempFile {
public:
  explicit ScopedTempFile(std::filesystem::path path) : path_(std::move(path)) {}
  ~ScopedTempFile() {
    std::error_code ec;
    std::filesystem::remove(path_, ec);
  }
  std::filesystem::path const& path() const {
    return path_;
  }

private:
  std::filesystem::path path_;
};

std::string read_text_file(std::filesystem::path const& path) {
  std::ifstream in(path);
  if (!in) {
    return "";
  }
  std::ostringstream out;
  out << in.rdbuf();
  return internal::trim_copy(out.str());
}

#ifdef _WIN32
std::wstring utf8_to_wide(std::string const& value) {
  if (value.empty()) {
    return std::wstring();
  }
  int const required =
      MultiByteToWideChar(CP_UTF8, 0, value.c_str(), static_cast<int>(value.size()), nullptr, 0);
  if (required <= 0) {
    return std::wstring();
  }
  std::wstring out(static_cast<size_t>(required), L'\0');
  if (MultiByteToWideChar(CP_UTF8, 0, value.c_str(), static_cast<int>(value.size()), out.data(),
                          required) <= 0) {
    return std::wstring();
  }
  return out;
}

std::string wide_to_utf8(std::wstring const& value) {
  if (value.empty()) {
    return std::string();
  }
  int const required = WideCharToMultiByte(
      CP_UTF8, 0, value.c_str(), static_cast<int>(value.size()), nullptr, 0, nullptr, nullptr);
  if (required <= 0) {
    return std::string();
  }
  std::string out(static_cast<size_t>(required), '\0');
  if (WideCharToMultiByte(CP_UTF8, 0, value.c_str(), static_cast<int>(value.size()), out.data(),
                          required, nullptr, nullptr) <= 0) {
    return std::string();
  }
  return out;
}

std::wstring quote_windows_arg(std::wstring const& arg) {
  if (arg.empty()) {
    return L"\"\"";
  }

  bool needs_quotes = false;
  for (wchar_t c : arg) {
    if (c == L' ' || c == L'\t' || c == L'"') {
      needs_quotes = true;
      break;
    }
  }
  if (!needs_quotes) {
    return arg;
  }

  std::wstring out;
  out.push_back(L'"');
  size_t backslashes = 0;
  for (wchar_t c : arg) {
    if (c == L'\\') {
      ++backslashes;
      continue;
    }
    if (c == L'"') {
      out.append(backslashes * 2 + 1, L'\\');
      out.push_back(L'"');
      backslashes = 0;
      continue;
    }
    out.append(backslashes, L'\\');
    backslashes = 0;
    out.push_back(c);
  }
  out.append(backslashes * 2, L'\\');
  out.push_back(L'"');
  return out;
}

std::wstring build_windows_command_line(std::vector<std::string> const& argv) {
  std::wstring out;
  for (size_t i = 0; i < argv.size(); ++i) {
    if (i > 0) {
      out.push_back(L' ');
    }
    out += quote_windows_arg(utf8_to_wide(argv[i]));
  }
  return out;
}
#endif

std::vector<std::string> split_command_line_portable(std::string const& cmd) {
  std::vector<std::string> parts;
  std::string current;
  bool in_quotes = false;
  bool escape = false;

  for (char c : cmd) {
    if (escape) {
      current.push_back(c);
      escape = false;
      continue;
    }
    if (c == '\\' && in_quotes) {
      escape = true;
      continue;
    }
    if (c == '"') {
      in_quotes = !in_quotes;
      continue;
    }
    if (!in_quotes && std::isspace(static_cast<unsigned char>(c)) != 0) {
      if (!current.empty()) {
        parts.push_back(current);
        current.clear();
      }
      continue;
    }
    current.push_back(c);
  }
  if (!current.empty()) {
    parts.push_back(current);
  }
  return parts;
}

RunnerCommand parse_runner_command(std::string const& raw) {
  RunnerCommand out;
  out.raw = internal::trim_copy(raw);
  if (out.raw.empty()) {
    out.error = "runner_not_configured";
    return out;
  }

#ifdef _WIN32
  std::wstring const wide = utf8_to_wide(out.raw);
  if (wide.empty()) {
    out.error = "runner_command_invalid";
    return out;
  }

  int argc = 0;
  LPWSTR* argv_w = CommandLineToArgvW(wide.c_str(), &argc);
  if (argv_w == nullptr || argc <= 0) {
    if (argv_w != nullptr) {
      LocalFree(static_cast<HLOCAL>(argv_w));
    }
    out.error = "runner_command_invalid";
    return out;
  }

  out.argv.reserve(static_cast<size_t>(argc));
  for (int i = 0; i < argc; ++i) {
    out.argv.push_back(wide_to_utf8(argv_w[i]));
  }
  LocalFree(static_cast<HLOCAL>(argv_w));
#else
  out.argv = split_command_line_portable(out.raw);
#endif

  if (out.argv.empty() || out.argv.front().empty()) {
    out.error = "runner_command_invalid";
  }
  return out;
}

std::string map_runner_exit_code(int exit_code) {
  switch (exit_code) {
    case 0:
      return "ok";
    case 10:
      return "runner_usage_error";
    case 11:
      return "runtime_unsupported";
    case 12:
      return "model_unavailable";
    case 13:
      return "input_read_failed";
    case 14:
      return "decode_failed";
    case 15:
      return "output_write_failed";
    case 16:
      return "invalid_output";
    default:
      return "runner_failed";
  }
}

ProcessResult run_process(std::vector<std::string> const& argv, int timeout_ms,
                          size_t output_limit) {
  ProcessResult result;
  if (argv.empty() || argv.front().empty()) {
    result.error = "runner_command_invalid";
    return result;
  }

  TailBuffer stdout_tail(output_limit);
  TailBuffer stderr_tail(output_limit);

#ifdef _WIN32
  SECURITY_ATTRIBUTES sa{};
  sa.nLength = sizeof(sa);
  sa.bInheritHandle = TRUE;
  sa.lpSecurityDescriptor = nullptr;

  HANDLE stdout_read = nullptr;
  HANDLE stdout_write = nullptr;
  HANDLE stderr_read = nullptr;
  HANDLE stderr_write = nullptr;

  if (!CreatePipe(&stdout_read, &stdout_write, &sa, 0) ||
      !CreatePipe(&stderr_read, &stderr_write, &sa, 0)) {
    if (stdout_read != nullptr)
      CloseHandle(stdout_read);
    if (stdout_write != nullptr)
      CloseHandle(stdout_write);
    if (stderr_read != nullptr)
      CloseHandle(stderr_read);
    if (stderr_write != nullptr)
      CloseHandle(stderr_write);
    result.error = "runner_pipe_create_failed";
    return result;
  }

  SetHandleInformation(stdout_read, HANDLE_FLAG_INHERIT, 0);
  SetHandleInformation(stderr_read, HANDLE_FLAG_INHERIT, 0);

  std::wstring command_line = build_windows_command_line(argv);
  std::vector<wchar_t> mutable_cmd(command_line.begin(), command_line.end());
  mutable_cmd.push_back(L'\0');

  STARTUPINFOW si{};
  si.cb = sizeof(si);
  si.dwFlags = STARTF_USESTDHANDLES;
  si.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
  si.hStdOutput = stdout_write;
  si.hStdError = stderr_write;

  PROCESS_INFORMATION pi{};
  BOOL created = CreateProcessW(nullptr, mutable_cmd.data(), nullptr, nullptr, TRUE,
                                CREATE_NO_WINDOW, nullptr, nullptr, &si, &pi);
  if (!created) {
    CloseHandle(stdout_read);
    CloseHandle(stdout_write);
    CloseHandle(stderr_read);
    CloseHandle(stderr_write);
    result.error = "runner_launch_failed";
    return result;
  }

  result.launched = true;

  CloseHandle(stdout_write);
  CloseHandle(stderr_write);

  auto read_pipe = [](HANDLE handle, TailBuffer* tail) {
    std::array<char, 1024> buffer{};
    DWORD bytes_read = 0;
    while (
        ReadFile(handle, buffer.data(), static_cast<DWORD>(buffer.size()), &bytes_read, nullptr) &&
        bytes_read > 0) {
      tail->append(buffer.data(), static_cast<size_t>(bytes_read));
    }
  };

  std::thread stdout_thread(read_pipe, stdout_read, &stdout_tail);
  std::thread stderr_thread(read_pipe, stderr_read, &stderr_tail);

  DWORD const wait_rc = WaitForSingleObject(pi.hProcess, static_cast<DWORD>(timeout_ms));
  if (wait_rc == WAIT_TIMEOUT) {
    result.timed_out = true;
    result.killed = (TerminateProcess(pi.hProcess, 124) != FALSE);
    WaitForSingleObject(pi.hProcess, 5000);
  } else if (wait_rc == WAIT_FAILED) {
    result.error = "runner_wait_failed";
  }

  DWORD exit_code = 0;
  if (!GetExitCodeProcess(pi.hProcess, &exit_code)) {
    result.error = "runner_exit_read_failed";
  } else {
    result.exit_code = static_cast<int>(exit_code);
  }

  CloseHandle(stdout_read);
  CloseHandle(stderr_read);
  if (stdout_thread.joinable()) {
    stdout_thread.join();
  }
  if (stderr_thread.joinable()) {
    stderr_thread.join();
  }

  CloseHandle(pi.hProcess);
  CloseHandle(pi.hThread);
#else
  int stdout_pipe[2];
  int stderr_pipe[2];
  if (pipe(stdout_pipe) != 0 || pipe(stderr_pipe) != 0) {
    result.error = "runner_pipe_create_failed";
    return result;
  }

  pid_t const pid = fork();
  if (pid < 0) {
    close(stdout_pipe[0]);
    close(stdout_pipe[1]);
    close(stderr_pipe[0]);
    close(stderr_pipe[1]);
    result.error = "runner_launch_failed";
    return result;
  }

  if (pid == 0) {
    dup2(stdout_pipe[1], STDOUT_FILENO);
    dup2(stderr_pipe[1], STDERR_FILENO);
    close(stdout_pipe[0]);
    close(stdout_pipe[1]);
    close(stderr_pipe[0]);
    close(stderr_pipe[1]);

    std::vector<char*> argv_raw;
    argv_raw.reserve(argv.size() + 1);
    for (std::string const& arg : argv) {
      argv_raw.push_back(const_cast<char*>(arg.c_str()));
    }
    argv_raw.push_back(nullptr);
    execvp(argv_raw[0], argv_raw.data());
    _exit(127);
  }

  result.launched = true;
  close(stdout_pipe[1]);
  close(stderr_pipe[1]);
  fcntl(stdout_pipe[0], F_SETFL, O_NONBLOCK);
  fcntl(stderr_pipe[0], F_SETFL, O_NONBLOCK);

  auto const deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
  int status = 0;
  while (true) {
    pollfd fds[2] = {
        {stdout_pipe[0], POLLIN, 0},
        {stderr_pipe[0], POLLIN, 0},
    };
    (void)poll(fds, 2, 20);

    std::array<char, 1024> buffer{};
    ssize_t bytes = read(stdout_pipe[0], buffer.data(), buffer.size());
    while (bytes > 0) {
      stdout_tail.append(buffer.data(), static_cast<size_t>(bytes));
      bytes = read(stdout_pipe[0], buffer.data(), buffer.size());
    }
    bytes = read(stderr_pipe[0], buffer.data(), buffer.size());
    while (bytes > 0) {
      stderr_tail.append(buffer.data(), static_cast<size_t>(bytes));
      bytes = read(stderr_pipe[0], buffer.data(), buffer.size());
    }

    pid_t const wait_rc = waitpid(pid, &status, WNOHANG);
    if (wait_rc == pid) {
      break;
    }

    if (std::chrono::steady_clock::now() >= deadline) {
      result.timed_out = true;
      result.killed = (kill(pid, SIGKILL) == 0);
      (void)waitpid(pid, &status, 0);
      break;
    }
  }

  close(stdout_pipe[0]);
  close(stderr_pipe[0]);
  if (WIFEXITED(status)) {
    result.exit_code = WEXITSTATUS(status);
  } else if (WIFSIGNALED(status)) {
    result.exit_code = 128 + WTERMSIG(status);
  }
#endif

  result.stdout_tail = internal::trim_copy(stdout_tail.data());
  result.stderr_tail = internal::trim_copy(stderr_tail.data());
  result.stdout_bytes = stdout_tail.total_bytes();
  result.stderr_bytes = stderr_tail.total_bytes();
  result.stdout_truncated = stdout_tail.truncated();
  result.stderr_truncated = stderr_tail.truncated();
  return result;
}

void apply_runner_process_metadata(nlohmann::json& meta, ProcessResult const& process) {
  meta["runner_exit_code"] = process.exit_code;
  meta["runner_stdout_tail"] = process.stdout_tail;
  meta["runner_stderr_tail"] = process.stderr_tail;
  meta["runner_stdout_bytes"] = process.stdout_bytes;
  meta["runner_stderr_bytes"] = process.stderr_bytes;
  meta["runner_stdout_truncated"] = process.stdout_truncated;
  meta["runner_stderr_truncated"] = process.stderr_truncated;
}

std::filesystem::path absolute_path(std::filesystem::path const& path) {
  if (path.empty()) {
    return path;
  }
  std::error_code ec;
  std::filesystem::path absolute = std::filesystem::absolute(path, ec);
  if (ec) {
    return path;
  }
  return absolute;
}

}  // namespace

std::string native_runtime_runner_env_name(std::string const& runtime_id) {
  std::string const runtime = internal::normalize_runtime_id(runtime_id);
  if (runtime == "tensorrt") {
    return "EARS_TENSORRT_ASR_RUNNER";
  }
  if (runtime == "openvino") {
    return "EARS_OPENVINO_ASR_RUNNER";
  }
  if (runtime == "coreml") {
    return "EARS_COREML_ASR_RUNNER";
  }
  if (runtime == "qnn") {
    return "EARS_QNN_ASR_RUNNER";
  }
  return "EARS_NATIVE_ASR_RUNNER";
}

bool is_native_runtime_id(std::string const& runtime_id) {
  std::string const runtime = internal::normalize_runtime_id(runtime_id);
  return runtime == "tensorrt" || runtime == "openvino" || runtime == "coreml" || runtime == "qnn";
}

NativeRuntimeRunnerProbe probe_native_runtime_runner(std::string const& runtime_id) {
  NativeRuntimeRunnerProbe probe;
  probe.runtime = internal::normalize_runtime_id(runtime_id);
  probe.runner_env = native_runtime_runner_env_name(probe.runtime);

  if (!is_native_runtime_id(probe.runtime)) {
    probe.status_code = "not_native_runtime";
    probe.message = "Runtime does not use external native runner.";
    return probe;
  }

  std::string const runner_cmd = trim_env(probe.runner_env);
  probe.runner_command = runner_cmd;
  if (runner_cmd.empty()) {
    probe.status_code = "runner_not_configured";
    probe.message = "Runner command env var is empty.";
    return probe;
  }

  RunnerCommand parsed = parse_runner_command(runner_cmd);
  if (!parsed.error.empty()) {
    probe.configured = true;
    probe.status_code = parsed.error;
    probe.message = "Unable to parse runner command.";
    return probe;
  }

  probe.configured = true;
  std::vector<std::string> argv = parsed.argv;
  argv.push_back("--probe");
  argv.push_back("--runtime");
  argv.push_back(probe.runtime);

  ProcessResult process = run_process(argv, resolve_runner_timeout_ms(probe.runtime, true),
                                      parse_output_tail_limit_bytes());
  probe.exit_code = process.exit_code;
  probe.timed_out = process.timed_out;
  probe.stdout_tail = process.stdout_tail;
  probe.stderr_tail = process.stderr_tail;
  probe.stdout_truncated = process.stdout_truncated;
  probe.stderr_truncated = process.stderr_truncated;

  if (!process.launched) {
    probe.status_code = process.error.empty() ? "runner_launch_failed" : process.error;
    probe.message = "Failed to launch runner process.";
    return probe;
  }
  if (process.timed_out) {
    probe.status_code = "runner_timeout";
    probe.message = "Runner probe timed out.";
    return probe;
  }
  if (process.exit_code != 0) {
    probe.status_code = map_runner_exit_code(process.exit_code);
    probe.message = "Runner probe returned non-zero exit code.";
    return probe;
  }

  probe.status_code = "ok";
  probe.message = "Runner probe succeeded.";
  probe.healthy = true;
  return probe;
}

class NativeRuntimeAsrAdapter::Impl {
public:
  Impl(std::string runtime_id, std::string family_id, std::string model_path, int sample_rate)
      : runtime_id_(internal::normalize_runtime_id(std::move(runtime_id))),
        family_id_(internal::to_lower_ascii(std::move(family_id))),
        model_path_(std::move(model_path)),
        sample_rate_(sample_rate) {}

  AsrResult recognize(float const* samples, size_t num_samples) {
    AsrResult out;
    nlohmann::json meta = {
        {"adapter", "native_runtime"}, {"runtime", runtime_id_},
        {"family", family_id_},        {"model_path", model_path_},
        {"sample_rate", sample_rate_}, {"runner_contract_version", kRunnerContractVersion},
    };

    if (samples == nullptr || num_samples == 0) {
      out.text = "";
      out.confidence = 0.0f;
      meta["decode"] = "empty_input";
      out.json = meta.dump();
      return out;
    }

    std::string const env_name = native_runtime_runner_env_name(runtime_id_);
    std::string const runner_cmd = trim_env(env_name);
    meta["runner_env"] = env_name;
    meta["runner_command"] = runner_cmd;

    if (runner_cmd.empty()) {
      out.text = "";
      out.confidence = 0.0f;
      meta["decode"] = "runner_not_configured";
      meta["code"] = "runner_not_configured";
      out.json = meta.dump();
      return out;
    }

    RunnerCommand parsed = parse_runner_command(runner_cmd);
    if (!parsed.error.empty()) {
      out.text = "";
      out.confidence = 0.0f;
      meta["decode"] = "runner_command_invalid";
      meta["code"] = parsed.error;
      out.json = meta.dump();
      return out;
    }

    std::filesystem::path temp_base = make_temp_base_path();
    ScopedTempFile input_f32(temp_base.string() + ".f32");
    ScopedTempFile output_txt(temp_base.string() + ".txt");
    std::filesystem::path const input_path = absolute_path(input_f32.path());
    std::filesystem::path const output_path = absolute_path(output_txt.path());
    std::filesystem::path const model_path = absolute_path(model_path_);

    bool wrote_input = false;
    {
      std::ofstream out_file(input_path, std::ios::binary | std::ios::trunc);
      if (out_file) {
        out_file.write(reinterpret_cast<char const*>(samples),
                       static_cast<std::streamsize>(num_samples * sizeof(float)));
        wrote_input = static_cast<bool>(out_file);
      }
    }
    if (!wrote_input) {
      out.text = "";
      out.confidence = 0.0f;
      meta["decode"] = "input_write_failed";
      meta["code"] = "input_write_failed";
      out.json = meta.dump();
      return out;
    }

    std::vector<std::string> argv = parsed.argv;
    argv.push_back("--runtime");
    argv.push_back(runtime_id_);
    argv.push_back("--family");
    argv.push_back(family_id_);
    argv.push_back("--model");
    argv.push_back(model_path.string());
    argv.push_back("--sample-rate");
    argv.push_back(std::to_string(sample_rate_));
    argv.push_back("--input-f32");
    argv.push_back(input_path.string());
    argv.push_back("--output-text");
    argv.push_back(output_path.string());

    int const timeout_ms = resolve_runner_timeout_ms(runtime_id_, false);
    meta["runner_timeout_ms"] = timeout_ms;

    ProcessResult process = run_process(argv, timeout_ms, parse_output_tail_limit_bytes());
    apply_runner_process_metadata(meta, process);

    if (!process.launched) {
      out.text = "";
      out.confidence = 0.0f;
      meta["decode"] = "runner_launch_failed";
      meta["code"] = process.error.empty() ? "runner_launch_failed" : process.error;
      out.json = meta.dump();
      return out;
    }
    if (process.timed_out) {
      out.text = "";
      out.confidence = 0.0f;
      meta["decode"] = "runner_timeout";
      meta["code"] = "runner_timeout";
      out.json = meta.dump();
      return out;
    }

    std::string const mapped_code = map_runner_exit_code(process.exit_code);
    if (process.exit_code != 0) {
      out.text = "";
      out.confidence = 0.0f;
      meta["decode"] = "runner_failed";
      meta["code"] = mapped_code;
      out.json = meta.dump();
      return out;
    }

    std::string const decoded = read_text_file(output_path);
    if (decoded.empty()) {
      out.text = "";
      out.confidence = 0.0f;
      meta["decode"] = "runner_empty_output";
      meta["code"] = "runner_empty_output";
      out.json = meta.dump();
      return out;
    }

    out.text = decoded;
    out.confidence = 0.9f;
    meta["decode"] = "ok";
    meta["code"] = "ok";
    out.json = meta.dump();
    return out;
  }

private:
  std::string runtime_id_;
  std::string family_id_;
  std::string model_path_;
  int sample_rate_ = 16000;
};

NativeRuntimeAsrAdapter::NativeRuntimeAsrAdapter(std::string runtime_id, std::string family_id,
                                                 std::string const& model_path, int sample_rate)
    : impl_(std::make_unique<Impl>(std::move(runtime_id), std::move(family_id), model_path,
                                   sample_rate)) {}

NativeRuntimeAsrAdapter::~NativeRuntimeAsrAdapter() = default;

AsrResult NativeRuntimeAsrAdapter::recognize(float const* samples, size_t num_samples) {
  return impl_->recognize(samples, num_samples);
}

}  // namespace ears
