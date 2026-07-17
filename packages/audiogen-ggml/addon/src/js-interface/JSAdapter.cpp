#include "js-interface/JSAdapter.hpp"

#include <optional>
#include <string>

#include "inference-addon-cpp/Errors.hpp"

namespace qvac::audiogenggml {

namespace js = qvac_lib_inference_addon_cpp::js;
namespace general_error = qvac_errors::general_error;

namespace {

std::optional<int> readOptionalInt(
    js::Object obj, js_env_t* env, const char* key) {
  js_value_t* raw = obj.getProperty(env, key);
  if (js::is<js::Undefined>(env, raw) || js::is<js::Null>(env, raw)) {
    return std::nullopt;
  }
  if (js::is<js::Number>(env, raw)) {
    return static_cast<int>(js::Number::fromValue(raw).as<double>(env));
  }
  if (js::is<js::String>(env, raw)) {
    const std::string str = js::String::fromValue(raw).as<std::string>(env);
    try {
      return std::stoi(str);
    } catch (const std::exception&) {
      throw qvac_errors::StatusError(
          general_error::InvalidArgument,
          std::string("Property '") + key +
              "' must be an integer (got \"" + str + "\")");
    }
  }
  throw qvac_errors::StatusError(
      general_error::InvalidArgument,
      std::string("Property '") + key + "' must be a number or numeric string");
}

std::optional<float> readOptionalFloat(
    js::Object obj, js_env_t* env, const char* key) {
  js_value_t* raw = obj.getProperty(env, key);
  if (js::is<js::Undefined>(env, raw) || js::is<js::Null>(env, raw)) {
    return std::nullopt;
  }
  if (js::is<js::Number>(env, raw)) {
    return static_cast<float>(js::Number::fromValue(raw).as<double>(env));
  }
  if (js::is<js::String>(env, raw)) {
    const std::string str = js::String::fromValue(raw).as<std::string>(env);
    try {
      return std::stof(str);
    } catch (const std::exception&) {
      throw qvac_errors::StatusError(
          general_error::InvalidArgument,
          std::string("Property '") + key +
              "' must be a number (got \"" + str + "\")");
    }
  }
  throw qvac_errors::StatusError(
      general_error::InvalidArgument,
      std::string("Property '") + key + "' must be a number or numeric string");
}

std::string readOptionalString(
    js::Object obj, js_env_t* env, const char* key) {
  auto v = obj.getOptionalPropertyAs<js::String, std::string>(env, key);
  return v.value_or(std::string{});
}

std::optional<bool> readOptionalBool(
    js::Object obj, js_env_t* env, const char* key) {
  return obj.getOptionalPropertyAs<js::Boolean, bool>(env, key);
}

}  // namespace

acestep::AcestepConfig JSAdapter::buildAcestepConfig(
    js::Object configurationParams, js_env_t* env) {
  acestep::AcestepConfig cfg;
  cfg.modelDir = readOptionalString(configurationParams, env, "modelDir");
  cfg.textEncModelPath =
      readOptionalString(configurationParams, env, "textEncModelPath");
  cfg.lmModelPath = readOptionalString(configurationParams, env, "lmModelPath");
  cfg.ditModelPath = readOptionalString(configurationParams, env, "ditModelPath");
  cfg.vaeModelPath = readOptionalString(configurationParams, env, "vaeModelPath");
  cfg.inferenceSteps = readOptionalInt(configurationParams, env, "inferenceSteps");
  cfg.shift = readOptionalFloat(configurationParams, env, "shift");
  cfg.seed = readOptionalInt(configurationParams, env, "seed");
  cfg.threads = readOptionalInt(configurationParams, env, "threads");
  cfg.useGpu = readOptionalBool(configurationParams, env, "useGPU");
  cfg.nGpuLayers = readOptionalInt(configurationParams, env, "nGpuLayers");
  cfg.backendsDir = readOptionalString(configurationParams, env, "backendsDir");
  return cfg;
}

}  // namespace qvac::audiogenggml
