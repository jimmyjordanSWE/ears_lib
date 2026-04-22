#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

namespace {

void print_shape(std::vector<int64_t> const& shape) {
  std::cout << "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i > 0) {
      std::cout << ", ";
    }
    std::cout << shape[i];
  }
  std::cout << "]";
}

void print_io(Ort::ConstSession const& session, bool inputs) {
  size_t const count = inputs ? session.GetInputCount() : session.GetOutputCount();
  std::cout << (inputs ? "inputs" : "outputs") << "=" << count << "\n";
  for (size_t i = 0; i < count; ++i) {
    std::string const name = inputs ? session.GetInputNameAllocated(i, Ort::AllocatorWithDefaultOptions()).get()
                                    : session.GetOutputNameAllocated(i, Ort::AllocatorWithDefaultOptions()).get();
    auto const info = inputs ? session.GetInputTypeInfo(i) : session.GetOutputTypeInfo(i);
    auto const tensor_info = info.GetTensorTypeAndShapeInfo();
    auto const shape = tensor_info.GetShape();
    std::cout << "  " << i << ": " << name << " ";
    print_shape(shape);
    std::cout << "\n";
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: ears_inspect_onnx_cli <model.onnx> [provider]\n";
    return 1;
  }

  try {
    std::string const model_path = argv[1];
    std::string const provider = argc >= 3 ? argv[2] : "cpu";
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ears_inspect");
    Ort::SessionOptions opts;
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
    if (provider == "cuda") {
      opts.AddConfigEntry("session.disable_cpu_ep_fallback", "1");
      OrtCUDAProviderOptions cuda_opts{};
      cuda_opts.device_id = 0;
      opts.AppendExecutionProvider_CUDA(cuda_opts);
    }
#if defined(_WIN32)
    std::wstring model_path_w(model_path.begin(), model_path.end());
    Ort::Session session(env, model_path_w.c_str(), opts);
#else
    Ort::Session session(env, model_path.c_str(), opts);
#endif
    std::cout << "model=" << model_path << "\n";
    std::cout << "provider=" << provider << "\n";
    print_io(session.GetConst(), true);
    print_io(session.GetConst(), false);
    return 0;
  } catch (Ort::Exception const& e) {
    std::cerr << "ONNX inspect failed: " << e.what() << "\n";
    return 2;
  } catch (std::exception const& e) {
    std::cerr << "inspect failed: " << e.what() << "\n";
    return 3;
  }
}
