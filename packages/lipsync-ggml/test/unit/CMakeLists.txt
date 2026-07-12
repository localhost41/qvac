find_package(GTest CONFIG REQUIRED)

add_executable(
  addon-test
  test_shapes.cpp
  test_parity.cpp
  ${CMAKE_SOURCE_DIR}/addon/src/model-interface/lam_audio2expression.cpp
  ${CMAKE_SOURCE_DIR}/addon/src/model-interface/model_factory.cpp
  ${CMAKE_SOURCE_DIR}/addon/src/utils/BackendSelection.cpp
  ${CMAKE_SOURCE_DIR}/addon/src/utils/LoggingMacros.cpp
)

target_compile_options(
  addon-test
  PRIVATE
    -Wno-deprecated
    -Wfatal-errors
    -g)

target_include_directories(
  addon-test
  PRIVATE
    ${CMAKE_SOURCE_DIR}/addon/src
    ${QVAC_LIB_INFERENCE_ADDON_CPP_INCLUDE_DIRS}
)

target_compile_features(addon-test PRIVATE cxx_std_20)

# Fixtures + default model location, so the parity test can find them from
# any working directory.
target_compile_definitions(
  addon-test
  PRIVATE
    LIPSYNC_FIXTURES_DIR="${CMAKE_SOURCE_DIR}/test/fixtures/reference"
    LIPSYNC_DEFAULT_GGUF="${CMAKE_SOURCE_DIR}/models/lam-audio2exp-f32.gguf")

target_link_libraries(
  addon-test
  PRIVATE
    ggml::ggml
    GTest::gtest
    GTest::gtest_main
    GTest::gmock_main
)

add_test(NAME LipsyncGgmlUnit COMMAND addon-test)
set_tests_properties(LipsyncGgmlUnit PROPERTIES
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  TIMEOUT 900)
