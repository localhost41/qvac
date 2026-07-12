import { LipsyncModel } from '@qvac/lipsync-ggml'
import {
  definePlugin,
  defineHandler,
  lipsyncConfigSchema,
  lipsyncRunRequestSchema,
  lipsyncRunResponseSchema,
  lipsyncHparamsRequestSchema,
  lipsyncHparamsResponseSchema,
  ModelType,
  ADDON_LIPSYNC,
  type CreateModelParams,
  type PluginModelResult,
  type LipsyncConfig
} from '@/schemas'
import { createStreamLogger, registerAddonLogger } from '@/logging'
import { lipsyncRun } from './ops/lipsync-run'
import { lipsyncGetHparams } from './ops/lipsync-hparams'

interface LipsyncLoadOptions {
  backend?: 'auto' | 'cpu'
}

interface LipsyncModelWrapper {
  load(force?: boolean): Promise<void>
  unload?(): Promise<void>
}

// The `@qvac/lipsync-ggml` LipsyncModel exposes `load({ backend })` rather
// than the `load(force?)` signature `PluginModel` expects. Wrap it so the
// plugin framework can call `load()` and have the configured backend flow
// through (same pattern as the VLA plugin).
function wrapLipsyncModel(
  inner: LipsyncModel,
  loadOpts: LipsyncLoadOptions
): LipsyncModel & LipsyncModelWrapper {
  const wrapper = inner as LipsyncModel & LipsyncModelWrapper
  const originalLoad = wrapper.load.bind(wrapper)
  wrapper.load = function load(): Promise<void> {
    return originalLoad(loadOpts)
  }
  return wrapper
}

export const lipsyncPlugin = definePlugin({
  modelType: ModelType.ggmlLipsync,
  displayName: 'Lipsync (LAM Audio2Expression ggml)',
  addonPackage: ADDON_LIPSYNC,
  loadConfigSchema: lipsyncConfigSchema,

  createModel(params: CreateModelParams): PluginModelResult {
    const config = (params.modelConfig ?? {}) as LipsyncConfig
    const logger = createStreamLogger(params.modelId, ModelType.ggmlLipsync)
    registerAddonLogger(params.modelId, ModelType.ggmlLipsync, logger)

    const inner = new LipsyncModel({
      files: { model: [params.modelPath] },
      ...(config.verbosity !== undefined && {
        config: { verbosity: config.verbosity }
      }),
      logger,
      opts: { stats: true }
    })

    const backend = config.backend ?? 'auto'
    const model = wrapLipsyncModel(inner, { backend })
    return { model }
  },

  handlers: {
    lipsyncRun: defineHandler({
      requestSchema: lipsyncRunRequestSchema,
      responseSchema: lipsyncRunResponseSchema,
      streaming: false,
      // The addon exposes a model-wide cancel() that interrupts the running
      // window inference; mirrors the VLA plugin's cancel surface.
      cancel: { scope: 'model', hard: true },
      handler: lipsyncRun
    }),
    lipsyncHparams: defineHandler({
      requestSchema: lipsyncHparamsRequestSchema,
      responseSchema: lipsyncHparamsResponseSchema,
      streaming: false,
      cancel: { scope: 'none' },
      handler: lipsyncGetHparams
    })
  }
})
