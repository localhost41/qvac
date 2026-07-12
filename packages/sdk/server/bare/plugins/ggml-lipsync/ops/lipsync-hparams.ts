import { getModel } from '@/server/bare/registry/model-registry'
import {
  type LipsyncHparamsRequest,
  type LipsyncHparamsResponse,
  lipsyncHparamsSchema
} from '@/schemas'

interface LipsyncModelLike {
  hparams: unknown
  backendName: string | null
}

export function lipsyncGetHparams(request: LipsyncHparamsRequest): Promise<LipsyncHparamsResponse> {
  const model = getModel(request.modelId) as unknown as LipsyncModelLike
  // Validate the addon-reported hparams against our schema so the wire
  // shape stays consistent even if the underlying addon changes.
  const parsed = lipsyncHparamsSchema.parse(model.hparams)
  return Promise.resolve({
    hparams: parsed,
    backendName: model.backendName ?? null
  })
}
