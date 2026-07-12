declare module '@qvac/lipsync-ggml' {
  export interface LipsyncFiles {
    /** Absolute path(s) to the lam-audio2exp GGUF; the first entry is used. */
    model: string[]
  }

  export interface LipsyncConfig {
    /** Native log verbosity: 0=ERROR, 1=WARNING, 2=INFO, 3=DEBUG, 4=OFF. */
    verbosity?: number
    /** Override the ggml backend plugin directory (defaults to ./prebuilds). */
    backendsDir?: string
  }

  export interface LipsyncHparams {
    sampleRate: number
    fps: number
    nCoeffs: number
    nIdentity: number
    windowFrames: number
    /** ARKit-52 blendshape names, in output order. */
    coeffNames: string[]
  }

  export interface LipsyncRunInput {
    /** 16 kHz mono PCM in [-1, 1]. */
    pcm: Float32Array
    /** Identity one-hot index, 0..nIdentity-1 (default 0). */
    idIdx?: number
  }

  export interface LipsyncRunResult {
    /** frameCount × nCoeffs coefficients, frame-major, sigmoid range [0,1]. */
    frames: Float32Array
    frameCount: number
    fps: number
    stats: Record<string, number> | null
  }

  export interface LipsyncStreamResult {
    frames: Float32Array
    frameCount: number
    /** Timeline position of the first frame in this batch. */
    startTimeMs: number
  }

  export class LipsyncModel {
    constructor(options: {
      files: LipsyncFiles
      config?: LipsyncConfig
      logger?: unknown
      opts?: { stats?: boolean }
    })

    load(options?: { backend?: 'auto' | 'cpu' }): Promise<void>
    run(input: LipsyncRunInput): Promise<LipsyncRunResult>
    runStreaming(
      pcmStream: AsyncIterable<Float32Array>,
      options?: { idIdx?: number }
    ): AsyncGenerator<LipsyncStreamResult>
    pause(): Promise<void>
    cancel(): Promise<void>
    unload(): Promise<void>
    getState(): { configLoaded: boolean; weightsLoaded: boolean }

    readonly hparams: LipsyncHparams | null
    readonly backendName: string | null
  }

  export const SAMPLE_RATE: number
  export const FPS: number
  export const WINDOW_FRAMES: number
  export const WINDOW_SAMPLES: number
  export function framesForSamples(numSamples: number): number
  export function int16ToFloat32(pcm: Int16Array): Float32Array

  export const ERR_CODES: Record<string, number>
  export class QvacErrorAddonLipsync extends Error {}

  export default LipsyncModel
}
