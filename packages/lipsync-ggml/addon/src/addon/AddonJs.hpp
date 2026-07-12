#pragma once

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <inference-addon-cpp/JsInterface.hpp>
#include <inference-addon-cpp/JsUtils.hpp>
#include <inference-addon-cpp/Logger.hpp>
#include <inference-addon-cpp/ModelInterfaces.hpp>
#include <inference-addon-cpp/addon/AddonJs.hpp>
#include <inference-addon-cpp/handlers/JsOutputHandlerImplementations.hpp>
#include <inference-addon-cpp/handlers/OutputHandler.hpp>
#include <inference-addon-cpp/queue/OutputCallbackJs.hpp>

#include "../utils/LoggingMacros.hpp"
#include "AddonCpp.hpp"

namespace qvac_lib_infer_lipsync_ggml {

namespace detail {

// Resolve the AddonJs instance handle (arg 0) to the underlying
// LipsyncModel; needed by the lipsync-specific accessors below because the
// framework only stores the model behind an IModel reference.
inline LipsyncModel&
lipsyncFromInstance(js_env_t* env, js_value_t* instanceHandle) {
  using namespace qvac_lib_inference_addon_cpp;
  auto& instance = JsInterface::getInstance(env, instanceHandle);
  auto* model = dynamic_cast<LipsyncModel*>(&instance.addonCpp->model.get());
  if (model == nullptr) {
    throw qvac_errors::StatusError(
        qvac_errors::general_error::InvalidArgument,
        "Instance handle does not refer to a LipsyncModel");
  }
  return *model;
}

// Copy a JS Float32Array into a std::vector<float>. process() runs on a
// worker thread after the JS callback returns, so the buffer must be owned.
inline std::vector<float> copyFloat32(js_env_t* env, js_value_t* jsArr) {
  float* data = nullptr;
  size_t len = 0;
  if (js_get_typedarray_info(
          env,
          jsArr,
          nullptr,
          reinterpret_cast<void**>(&data),
          &len,
          nullptr,
          nullptr) != 0) {
    throw std::runtime_error("expected Float32Array");
  }
  return {data, data + len};
}

// Parse the run input object emitted by index.js: { pcm, idIdx }.
inline LipsyncInput parseRunInput(js_env_t* env, js_value_t* inputVal) {
  using namespace qvac_lib_inference_addon_cpp;
  js::Object obj(env, inputVal);

  LipsyncInput in;
  in.pcm =
      copyFloat32(env, obj.getProperty<js::TypedArray<float>>(env, "pcm"));
  in.idIdx = static_cast<uint32_t>(
      obj.getPropertyAs<js::Number, int32_t>(env, "idIdx"));
  return in;
}

} // namespace detail

// createInstance(jsHandle, { ggufPath, backend, backendsDir }, outputCb)
//   -> External
inline js_value_t* createInstance(js_env_t* env, js_callback_info_t* info) try {
  using namespace qvac_lib_inference_addon_cpp;

  JsArgsParser args(env, info);

  const std::string ggufPath = args.getMapEntry(1, "ggufPath");
  const std::string backend = args.getMapEntry(1, "backend");
  const std::string backendsDir = args.getMapEntry(1, "backendsDir");
  const bool forceCpu = (backend == "cpu");

  auto model = std::make_unique<LipsyncModel>(ggufPath, forceCpu, backendsDir);

  // Lipsync emits a single Float32Array (frames × 52 coefficients) per job;
  // runtime stats and errors are added by OutputCallBackJs.
  out_handl::OutputHandlers<out_handl::JsOutputHandlerInterface> outHandlers;
  outHandlers.add(
      std::make_shared<out_handl::JsTypedArrayOutputHandler<float>>());
  std::unique_ptr<OutputCallBackInterface> callback =
      std::make_unique<OutputCallBackJs>(
          env,
          args.get(0, "jsHandle"),
          args.getFunction(2, "outputCallback"),
          std::move(outHandlers));

  auto addon =
      std::make_unique<AddonJs>(env, std::move(callback), std::move(model));
  return JsInterface::createInstance(env, std::move(addon));
}
JSCATCH

// runJob(instance, { type: 'lipsync', input: { pcm, idIdx } }) -> bool
//
// Returns true if the job was accepted, false if a previous job is still in
// flight. Output (Float32Array frames) and stats arrive asynchronously on
// the outputCb registered at createInstance.
inline js_value_t* runJob(js_env_t* env, js_callback_info_t* info) try {
  using namespace qvac_lib_inference_addon_cpp;

  JsArgsParser args(env, info);
  auto [type, jsInput] = JsInterface::getInput(args);
  if (type != "lipsync") {
    throw qvac_errors::StatusError(
        qvac_errors::general_error::InvalidArgument,
        "Unknown input type: " + type);
  }

  std::any input{detail::parseRunInput(env, jsInput)};
  return JsInterface::getInstance(env, args.get(0, "instance"))
      .runJob(std::move(input));
}
JSCATCH

// getLipsyncBackendName(instance) -> string ("CPU", …)
inline js_value_t*
getLipsyncBackendName(js_env_t* env, js_callback_info_t* info) try {
  using namespace qvac_lib_inference_addon_cpp;

  JsArgsParser args(env, info);
  LipsyncModel& model =
      detail::lipsyncFromInstance(env, args.get(0, "instance"));
  const std::string name = model.backendName();

  js_value_t* str = nullptr;
  if (js_create_string_utf8(
          env,
          reinterpret_cast<const utf8_t*>(name.c_str()),
          name.size(),
          &str) != 0) {
    throw std::runtime_error("js_create_string_utf8 failed");
  }
  return str;
}
JSCATCH

// getLipsyncHparams(instance) -> { sampleRate, fps, nCoeffs, nIdentity,
//                                  windowFrames, coeffNames }
//
// Everything JS-side input validation and downstream consumers (renderer
// blendshape mapping) need; coeffNames is the ARKit-52 order table baked
// into the GGUF.
inline js_value_t*
getLipsyncHparams(js_env_t* env, js_callback_info_t* info) try {
  using namespace qvac_lib_inference_addon_cpp;

  JsArgsParser args(env, info);
  LipsyncModel& model =
      detail::lipsyncFromInstance(env, args.get(0, "instance"));
  const LamHParams& hp = model.hparams();

  js_value_t* obj = nullptr;
  if (js_create_object(env, &obj) != 0) {
    throw std::runtime_error("js_create_object failed");
  }
  auto setInt = [&](const char* name, int32_t value) {
    js_value_t* v = nullptr;
    js_create_int32(env, value, &v);
    js_set_named_property(env, obj, name, v);
  };
  setInt("sampleRate", static_cast<int32_t>(hp.sampleRate));
  setInt("fps", static_cast<int32_t>(hp.fps));
  setInt("nCoeffs", static_cast<int32_t>(hp.nCoeffs));
  setInt("nIdentity", static_cast<int32_t>(hp.nIdentity));
  setInt("windowFrames", static_cast<int32_t>(hp.windowFrames));

  js_value_t* names = nullptr;
  if (js_create_array_with_length(env, hp.coeffNames.size(), &names) != 0) {
    throw std::runtime_error("js_create_array_with_length failed");
  }
  for (size_t i = 0; i < hp.coeffNames.size(); ++i) {
    js_value_t* s = nullptr;
    js_create_string_utf8(
        env,
        reinterpret_cast<const utf8_t*>(hp.coeffNames[i].c_str()),
        hp.coeffNames[i].size(),
        &s);
    js_set_element(env, names, static_cast<uint32_t>(i), s);
  }
  js_set_named_property(env, obj, "coeffNames", names);
  return obj;
}
JSCATCH

// setVerbosity(level: 0..4) -> undefined
//
// 0=ERROR, 1=WARNING, 2=INFO, 3=DEBUG, 4=OFF (matches @qvac/logging
// priorities). Out-of-range values clamp to ERROR.
inline js_value_t* setVerbosity(js_env_t* env, js_callback_info_t* info) try {
  using namespace qvac_lib_inference_addon_cpp;
  using Priority = qvac_lib_inference_addon_cpp::logger::Priority;

  JsArgsParser args(env, info);
  const int32_t level = js::Number(env, args.get(0, "level")).as<int32_t>(env);
  Priority p = Priority::ERROR;
  if (level >= 0 && level <= static_cast<int32_t>(Priority::OFF)) {
    p = static_cast<Priority>(level);
  }
  qvac_lib_infer_lipsync_ggml::logging::g_verbosityLevel.store(
      p, std::memory_order_relaxed);

  js_value_t* undef = nullptr;
  js_get_undefined(env, &undef);
  return undef;
}
JSCATCH

} // namespace qvac_lib_infer_lipsync_ggml
