#include <any>
#include <chrono>
#include <memory>
#include <string>
#include <thread>

#include <bare.h>
#include <js.h>

#include "inference-addon-cpp/JsInterface.hpp"
#include "inference-addon-cpp/JsUtils.hpp"
#include "inference-addon-cpp/ModelInterfaces.hpp"
#include "inference-addon-cpp/addon/AddonJs.hpp"
#include "inference-addon-cpp/handlers/JsOutputHandlerImplementations.hpp"
#include "inference-addon-cpp/queue/OutputCallbackJs.hpp"

namespace {

namespace addon_cpp = qvac_lib_inference_addon_cpp;
namespace js = qvac_lib_inference_addon_cpp::js;

class EchoModel : public addon_cpp::model::IModel {
public:
  std::string getName() const override { return "EchoModel"; }

  std::any process(const std::any& input) override {
    return std::any_cast<std::string>(input);
  }

  addon_cpp::RuntimeStats runtimeStats() const override { return {}; }
};

js_value_t* createInstance(js_env_t* env, js_callback_info_t* info) try {
  addon_cpp::JsArgsParser args(env, info);

  addon_cpp::out_handl::OutputHandlers<
      addon_cpp::out_handl::JsOutputHandlerInterface>
      outputHandlers;
  outputHandlers.add(
      std::make_shared<addon_cpp::out_handl::JsStringOutputHandler>());

  auto outputCallback = std::make_unique<addon_cpp::OutputCallBackJs>(
      env,
      args.get(0, "jsHandle"),
      args.getFunction(1, "outputCallback"),
      std::move(outputHandlers));

  auto addon = std::make_unique<addon_cpp::AddonJs>(
      env, std::move(outputCallback), std::make_unique<EchoModel>());

  return addon_cpp::JsInterface::createInstance(env, std::move(addon));
}
JSCATCH

js_value_t* runJob(js_env_t* env, js_callback_info_t* info) try {
  addon_cpp::JsArgsParser args(env, info);
  auto& instance =
      addon_cpp::JsInterface::getInstance(env, args.get(0, "instance"));
  auto input = js::String(env, args.get(1, "input")).as<std::string>(env);
  instance.addonCpp->runJob(std::any(std::move(input)));
  return nullptr;
}
JSCATCH

js_value_t* blockEventLoop(js_env_t* env, js_callback_info_t* info) try {
  addon_cpp::JsArgsParser args(env, info);
  const auto ms =
      js::Number(env, args.get(0, "milliseconds")).as<int32_t>(env);
  std::this_thread::sleep_for(std::chrono::milliseconds(ms));
  return nullptr;
}
JSCATCH

// QVAC-21914: reproduce the cancel()+unload() teardown-thread race. Mirrors the
// llm-llamacpp cancel() path — capture the AddonCpp by shared_ptr into a
// JsAsyncTask so it outlives a racing destroyInstance(). The short worker sleep
// deterministically lets the JS-thread destroyInstance() drop its reference
// first, so the detached worker holds the LAST shared_ptr. Before the fix, the
// worker then destroyed ~AddonCpp/~OutputCallBackJs (js_delete_reference /
// uv_close) off the JS thread and aborted the runtime; after the fix the work
// functor is released in onComplete on the JS thread.
js_value_t* cancel(js_env_t* env, js_callback_info_t* info) try {
  addon_cpp::JsArgsParser args(env, info);
  auto& instance =
      addon_cpp::JsInterface::getInstance(env, args.get(0, "instance"));
  auto addonCppRef = instance.addonCpp;
  return js::JsAsyncTask::run(env, [addonCppRef]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    addonCppRef->cancelJob();
  });
}
JSCATCH

js_value_t* outputCallbackLifetimeExports(js_env_t* env, js_value_t* exports) {
#define V(name, fn)                                                            \
  {                                                                            \
    js_value_t* val;                                                           \
    if (js_create_function(env, name, -1, fn, nullptr, &val) != 0) {           \
      return nullptr;                                                          \
    }                                                                          \
    if (js_set_named_property(env, exports, name, val) != 0) {                 \
      return nullptr;                                                          \
    }                                                                          \
  }

  V("createInstance", createInstance)
  V("runJob", runJob)
  V("blockEventLoop", blockEventLoop)
  V("cancel", cancel)
  V("destroyInstance", addon_cpp::JsInterface::destroyInstance)
#undef V

  return exports;
}

} // namespace

BARE_MODULE(output_callback_lifetime, outputCallbackLifetimeExports)
