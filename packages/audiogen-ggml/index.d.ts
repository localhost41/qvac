export const ENGINE_ACESTEP: 'acestep'

export interface AudioGenOptions {
  modelDir?: string
  textEncModel?: string
  lmModel?: string
  ditModel?: string
  vaeModel?: string
  inferenceSteps?: number
  shift?: number
  useGpu?: boolean
  threads?: number
}

export interface GenerateOptions {
  lyrics?: string
  seed?: number
  vocalLanguage?: string
}

export interface GenerateResult {
  outputArray: Int16Array
  sampleRate: number
  channels: number
  metadata: {
    caption?: string
    lyrics?: string
    keyscale?: string
    bpm?: number
    timesignature?: number
    vocalLanguage?: string
    seed?: number
    codes?: number
  }
}

export type OutputCallback = (event: unknown) => void

export class AudioGen {
  constructor (options?: AudioGenOptions, outputCb?: OutputCallback | null)
  activate (): Promise<void>
  generate (caption: string, opts?: GenerateOptions): Promise<GenerateResult>
  cancel (): Promise<void>
  destroy (): Promise<void>
  unload (): Promise<void>
}
