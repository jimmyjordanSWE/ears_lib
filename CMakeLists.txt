cmake_minimum_required(VERSION 3.16)
project(ears_lib VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
file(TO_CMAKE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/config/master.json" EARS_MASTER_CONFIG_PATH)

option(EARS_USE_CUDA "Enable CUDA provider for ONNX Runtime" OFF)
option(EARS_NO_ONNX "Build without ONNX (skeleton/demo only, no VAD/ASR/LLM)" OFF)
option(EARS_SANITIZE "Build with AddressSanitizer and UndefinedBehaviorSanitizer (Clang only)" OFF)
option(EARS_ENABLE_TENSORRT_BACKEND "Enable TensorRT backend registration" ON)
option(EARS_ENABLE_OPENVINO_BACKEND "Enable OpenVINO backend registration" ON)
option(EARS_ENABLE_COREML_BACKEND "Enable CoreML backend registration" ON)
option(EARS_ENABLE_QNN_BACKEND "Enable QNN backend registration" ON)
set(EARS_WARNING_PROFILE "medium" CACHE STRING "Compiler warning profile (low, medium, strict, max)")
set_property(CACHE EARS_WARNING_PROFILE PROPERTY STRINGS low medium strict max)

set(EARS_HAS_ONNX OFF)
set(EARS_HAS_TENSORRT ${EARS_ENABLE_TENSORRT_BACKEND})
set(EARS_HAS_OPENVINO ${EARS_ENABLE_OPENVINO_BACKEND})
set(EARS_HAS_COREML ${EARS_ENABLE_COREML_BACKEND})
set(EARS_HAS_QNN ${EARS_ENABLE_QNN_BACKEND})

if(NOT EARS_NO_ONNX)
  set(ONNXRUNTIME_ROOT "" CACHE PATH "Path to ONNX Runtime install (include + lib)")
  if(DEFINED ENV{ONNXRUNTIME_ROOT} AND NOT ONNXRUNTIME_ROOT)
    set(ONNXRUNTIME_ROOT $ENV{ONNXRUNTIME_ROOT} CACHE PATH "" FORCE)
  endif()

  find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
    HINTS ${ONNXRUNTIME_ROOT}
    PATH_SUFFIXES include
  )

  find_library(ONNXRUNTIME_LIBRARY
    NAMES onnxruntime
    HINTS ${ONNXRUNTIME_ROOT}
    PATH_SUFFIXES lib lib64
  )

  if(EARS_USE_CUDA)
    find_library(ONNXRUNTIME_CUDA_LIBRARY
      NAMES onnxruntime_providers_cuda
      HINTS ${ONNXRUNTIME_ROOT}
      PATH_SUFFIXES lib lib64
    )
  endif()

  if(ONNXRUNTIME_INCLUDE_DIR AND ONNXRUNTIME_LIBRARY)
    set(EARS_HAS_ONNX ON)
  else()
    message(STATUS "ONNX Runtime not found. Building skeleton only (EARS_NO_ONNX).")
  endif()
endif()

find_package(nlohmann_json QUIET)
if(NOT nlohmann_json_FOUND)
  include(FetchContent)
  FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
  )
  FetchContent_MakeAvailable(nlohmann_json)
endif()

function(ears_apply_warning_profile target_name)
  if(MSVC)
    target_compile_options(${target_name} PRIVATE /EHsc)
    if(EARS_WARNING_PROFILE STREQUAL "low")
      target_compile_options(${target_name} PRIVATE /W3)
    elseif(EARS_WARNING_PROFILE STREQUAL "medium")
      target_compile_options(${target_name} PRIVATE /W4)
    elseif(EARS_WARNING_PROFILE STREQUAL "strict")
      target_compile_options(${target_name} PRIVATE
        /W4 /permissive-
        /w14242 /w14254 /w14263 /w14265 /w14287
        /w14289 /w14296 /w14311
      )
    elseif(EARS_WARNING_PROFILE STREQUAL "max")
      target_compile_options(${target_name} PRIVATE
        /W4 /permissive-
        /w14242 /w14254 /w14263 /w14265 /w14287
        /w14289 /w14296 /w14311
        /w14545 /w14546 /w14547 /w14549 /w14555
        /w14619 /w14640 /w14826
        /w14905 /w14906 /w14928
      )
    endif()
  else()
    if(EARS_WARNING_PROFILE STREQUAL "low")
      target_compile_options(${target_name} PRIVATE -Wall)
    elseif(EARS_WARNING_PROFILE STREQUAL "medium")
      target_compile_options(${target_name} PRIVATE -Wall -Wextra)
    elseif(EARS_WARNING_PROFILE STREQUAL "strict")
      target_compile_options(${target_name} PRIVATE -Wall -Wextra -Wpedantic)
    elseif(EARS_WARNING_PROFILE STREQUAL "max")
      target_compile_options(${target_name} PRIVATE
        -Wall -Wextra -Wpedantic
        -Wconversion -Wsign-conversion -Wshadow -Wdouble-promotion
      )
    endif()
  endif()
endfunction()

function(ears_apply_sanitizers target_name)
  if(NOT EARS_SANITIZE)
    return()
  endif()

  if(MSVC)
    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
      target_compile_options(${target_name} PRIVATE /fsanitize=address /Zi)
      target_link_options(${target_name} PRIVATE /DEBUG /INCREMENTAL:NO)
    else()
      message(STATUS "EARS_SANITIZE=ON but MSVC ASan is only enabled for 64-bit targets")
    endif()
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(${target_name} PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(${target_name} PRIVATE -fsanitize=address,undefined)
  else()
    message(STATUS "EARS_SANITIZE=ON but compiler does not have configured sanitizer flags")
  endif()
endfunction()

set(EARS_SOURCES
  src/config.cpp
  src/factory.cpp
  src/factory_backends.cpp
  src/asr/coreml_asr_adapter.cpp
  src/asr/native_runtime_asr_adapter.cpp
  src/asr/openvino_asr_adapter.cpp
  src/asr/qnn_asr_adapter.cpp
  src/asr/tensorrt_asr_adapter.cpp
  src/mel.cpp
  src/pipeline.cpp
  src/transcribe.cpp
)
if(EARS_HAS_ONNX)
  list(APPEND EARS_SOURCES
    src/asr/ctc_asr_adapter.cpp
    src/asr/hybrid_ctc_attention_adapter.cpp
    src/asr/moonshine_asr.cpp
    src/asr/rnnt_asr_adapter.cpp
    src/asr/streaming_seq2seq_adapter.cpp
    src/asr/whisper_asr.cpp
    src/asr/whisper_seq2seq_adapter.cpp
    src/llm/smollm_rectifier.cpp
    src/vad/silero_vad.cpp
  )
endif()
add_library(ears STATIC ${EARS_SOURCES})
ears_apply_warning_profile(ears)
ears_apply_sanitizers(ears)

target_include_directories(ears PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  $<$<BOOL:${EARS_HAS_ONNX}>:${ONNXRUNTIME_INCLUDE_DIR}>
)

target_link_libraries(ears PUBLIC
  nlohmann_json::nlohmann_json
  $<$<BOOL:${EARS_HAS_ONNX}>:${ONNXRUNTIME_LIBRARY}>
)

if(EARS_HAS_ONNX AND EARS_USE_CUDA AND ONNXRUNTIME_CUDA_LIBRARY)
  target_link_libraries(ears PUBLIC ${ONNXRUNTIME_CUDA_LIBRARY})
  target_compile_definitions(ears PRIVATE EARS_USE_CUDA=1)
endif()

target_compile_definitions(ears PRIVATE
  EARS_HAS_ONNX=$<BOOL:${EARS_HAS_ONNX}>
  EARS_HAS_TENSORRT=$<BOOL:${EARS_HAS_TENSORRT}>
  EARS_HAS_OPENVINO=$<BOOL:${EARS_HAS_OPENVINO}>
  EARS_HAS_COREML=$<BOOL:${EARS_HAS_COREML}>
  EARS_HAS_QNN=$<BOOL:${EARS_HAS_QNN}>
  EARS_MASTER_CONFIG_PATH="${EARS_MASTER_CONFIG_PATH}"
)

option(EARS_BUILD_TESTS "Build test suite" ON)

if(EARS_BUILD_TESTS)
  include(FetchContent)
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
  )
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)

  if(EARS_SANITIZE)
    if(TARGET gtest)
      ears_apply_sanitizers(gtest)
    endif()
    if(TARGET gtest_main)
      ears_apply_sanitizers(gtest_main)
    endif()
    if(TARGET gmock)
      ears_apply_sanitizers(gmock)
    endif()
    if(TARGET gmock_main)
      ears_apply_sanitizers(gmock_main)
    endif()
  endif()

  enable_testing()

  add_executable(ears_test
    tests/config_test.cpp
    tests/factory_test.cpp
    tests/native_runtime_runner_test.cpp
    tests/onnx_asr_common_test.cpp
    tests/pipeline_test.cpp
    tests/types_test.cpp
  )
  ears_apply_warning_profile(ears_test)
  ears_apply_sanitizers(ears_test)

  target_include_directories(ears_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/tests
  )
  target_link_libraries(ears_test PRIVATE
    ears
    GTest::gtest_main
  )
  target_compile_definitions(ears_test PRIVATE
    EARS_HAS_ONNX=0
    EARS_EXPECT_HAS_ONNX=$<BOOL:${EARS_HAS_ONNX}>
    EARS_EXPECT_HAS_TENSORRT=$<BOOL:${EARS_HAS_TENSORRT}>
    EARS_EXPECT_HAS_OPENVINO=$<BOOL:${EARS_HAS_OPENVINO}>
    EARS_EXPECT_HAS_COREML=$<BOOL:${EARS_HAS_COREML}>
    EARS_EXPECT_HAS_QNN=$<BOOL:${EARS_HAS_QNN}>
  )

  include(GoogleTest)
  gtest_discover_tests(ears_test)
endif()
