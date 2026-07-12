cmake_minimum_required(VERSION 3.25)

option(ANDROID_STL "Android STL linkage" c++_shared)
option(BUILD_TESTING "Build tests" OFF)
option(ENABLE_COVERAGE "Enable coverage instrumentation for unit tests" OFF)
if(BUILD_TESTING)
  list(APPEND VCPKG_MANIFEST_FEATURES "tests")
endif()

find_package(cmake-bare REQUIRED PATHS node_modules/cmake-bare)
find_package(cmake-vcpkg REQUIRED PATHS node_modules/cmake-vcpkg)

set(VCPKG_OVERLAY_TRIPLETS "${CMAKE_CURRENT_SOURCE_DIR}/../../vcpkg-overlays/triplets;${VCPKG_OVERLAY_TRIPLETS}")

project(qvac-lib-infer-lipsync C CXX)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  add_compile_options(-stdlib=libc++)
  add_link_options(-stdlib=libc++ -static-libstdc++)
endif()

find_path(VCPKG_INSTALLED_PATH share/lint-cpp/.clang-format REQUIRED)
configure_file(${VCPKG_INSTALLED_PATH}/share/lint-cpp/.clang-format
               ${CMAKE_CURRENT_SOURCE_DIR}/.clang-format COPYONLY)
configure_file(${VCPKG_INSTALLED_PATH}/share/lint-cpp/.clang-tidy
               ${CMAKE_CURRENT_SOURCE_DIR}/.clang-tidy COPYONLY)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_path(QVAC_LIB_INFERENCE_ADDON_CPP_INCLUDE_DIRS "inference-addon-cpp/JsInterface.hpp" REQUIRED)

# ggml is provided by the qvac-fabric vcpkg port (same port that ships
# llama.cpp for the LLM addon). Backends (Vulkan, Metal, …) are loaded at
# runtime via ggml_backend_load_all_from_path; here we just link the core
# libraries and install each available backend as a shared library.
find_package(ggml CONFIG REQUIRED)

if(WIN32)
  add_definitions(-DNOMINMAX -DWIN32_LEAN_AND_MEAN -DNOGDI)
endif()

# Compute the per-addon "backends" directory that the module will ship; the
# C++ code uses this path at runtime to discover backend plugins (mirrors the
# LLM / VLA addons).
bare_target(bare_target_value)
bare_module_target("." unused_target NAME module_name VERSION unused_version)
set(BACKENDS_SUBDIR_VALUE "${bare_target_value}/${module_name}")
message("Building qvac-lib-infer-lipsync with BACKENDS_SUBDIR='${BACKENDS_SUBDIR_VALUE}'")

# Collect backend shared libraries to export alongside the .bare module.
set(BACKEND_DL_LIBS "")
if((ANDROID OR UNIX) AND NOT APPLE)
  foreach(_backend ${GGML_AVAILABLE_BACKENDS})
    list(APPEND BACKEND_DL_LIBS INSTALL TARGET ggml::${_backend})
  endforeach()
endif()

add_bare_module(qvac-lib-infer-lipsync EXPORTS ${BACKEND_DL_LIBS})

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  target_link_options(${qvac-lib-infer-lipsync}_module PRIVATE -Wl,--exclude-libs,ALL)
endif()

target_sources(
  ${qvac-lib-infer-lipsync}
  PRIVATE
    ${PROJECT_SOURCE_DIR}/addon/src/js-interface/binding.cpp
    ${PROJECT_SOURCE_DIR}/addon/src/model-interface/lam_audio2expression.cpp
    ${PROJECT_SOURCE_DIR}/addon/src/model-interface/model_factory.cpp
    ${PROJECT_SOURCE_DIR}/addon/src/utils/BackendSelection.cpp
    ${PROJECT_SOURCE_DIR}/addon/src/utils/LoggingMacros.cpp
)

target_include_directories(
  ${qvac-lib-infer-lipsync}
  PRIVATE
    ${QVAC_LIB_INFERENCE_ADDON_CPP_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/addon/src
)

target_link_libraries(${qvac-lib-infer-lipsync} PRIVATE ggml::ggml)

# Route QLOG calls into the JS-side logger installed via setLogger().
target_compile_definitions(${qvac-lib-infer-lipsync} PUBLIC JS_LOGGER)

target_compile_definitions(${qvac-lib-infer-lipsync}
  PRIVATE
    BACKENDS_SUBDIR="${BACKENDS_SUBDIR_VALUE}")

if(GGML_BACKEND_DL)
  target_compile_definitions(${qvac-lib-infer-lipsync} PRIVATE GGML_BACKEND_DL)
endif()

if(BUILD_TESTING)
  find_package(GTest CONFIG REQUIRED)
  include(GoogleTest)
  enable_testing()

  add_subdirectory(test/unit)
endif()
