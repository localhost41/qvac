'use strict'

const fs = require('bare-fs')
const path = require('bare-path')
const QvacLogger = require('@qvac/logging')
const { createJobHandler, exclusiveRunQueue } = require('@qvac/infer-base')
const binding = require('./binding')
const {
  SAMPLE_RATE, FPS, WINDOW_FRAMES, WINDOW_SAMPLES,
  framesForSamples, int16ToFloat32, StreamingWindow
} = require('./addon.js')
const { QvacErrorAddonLipsync, ERR_CODES } = require('./lib/error')

// Maps the C++ Priority enum (0=ERROR, 1=WARNING, 2=INFO, 3=DEBUG) to the
// matching method on the JS QvacLogger instance. Mirrors vla-ggml.
const LOG_METHODS = ['error', 'warn', 'info', 'debug']

const DEFAULT_NATIVE_VERBOSITY = 2 // INFO

function validateRunInput (input, hparams) {
  if (!input || typeof input !== 'object') {
    throw new QvacErrorAddonLipsync({ code: ERR_CODES.INVALID_INPUT, adds: 'input must be an object' })
  }
  if (!(input.pcm instanceof Float32Array) || input.pcm.length === 0) {
    throw new QvacErrorAddonLipsync({ code: ERR_CODES.INVALID_INPUT, adds: 'input.pcm must be a non-empty Float32Array of 16 kHz mono samples' })
  }
  // The graph needs at least 2 output frames / 2 conv-extractor steps.
  const minSamples = Math.ceil(2 * SAMPLE_RATE / FPS) + 400
  if (input.pcm.length < minSamples) {
    throw new QvacErrorAddonLipsync({ code: ERR_CODES.INVALID_INPUT, adds: `input.pcm too short (${input.pcm.length} samples; need >= ${minSamples})` })
  }
  const idIdx = input.idIdx ?? 0
  const nIdentity = (hparams && Number.isInteger(hparams.nIdentity)) ? hparams.nIdentity : 12
  if (!Number.isInteger(idIdx) || idIdx < 0 || idIdx >= nIdentity) {
    throw new QvacErrorAddonLipsync({ code: ERR_CODES.INVALID_INPUT, adds: `input.idIdx must be an integer in [0, ${nIdentity})` })
  }
  return { idIdx }
}

class LipsyncModel {
  constructor ({ files, config = {}, logger = null, opts = {} } = {}) {
    if (!files || !Array.isArray(files.model) || files.model.length === 0) {
      throw new QvacErrorAddonLipsync({ code: ERR_CODES.MISSING_REQUIRED_PARAMETER, adds: 'files.model (non-empty array of absolute paths)' })
    }
    for (const [i, entry] of files.model.entries()) {
      if (typeof entry !== 'string' || entry.length === 0) {
        throw new QvacErrorAddonLipsync({ code: ERR_CODES.INVALID_CONFIG, adds: `files.model[${i}] must be an absolute path string` })
      }
      if (!path.isAbsolute(entry)) {
        throw new QvacErrorAddonLipsync({ code: ERR_CODES.INVALID_CONFIG, adds: `files.model[${i}] must be an absolute path (got: ${entry})` })
      }
    }
    this._files = files.model
    this._config = config
    this.logger = new QvacLogger(logger)
    this.opts = opts
    this._job = createJobHandler({ cancel: () => this.cancel() })
    this._run = exclusiveRunQueue()
    this._handle = null
    this._hparams = null
    this._backendName = null
    this._hasActiveResponse = false
    this._nativeLoggerActive = false
    this._packageName = '@qvac/lipsync-ggml'
    this._packageVersion = require('./package.json').version
    // Per-run accumulator filled by _onAddonEvent; null between runs.
    this._pending = null
    this.state = { configLoaded: false, weightsLoaded: false }
  }

  _connectNativeLogger () {
    if (this._nativeLoggerActive) return
    try {
      binding.setLogger((priority, message) => {
        const method = LOG_METHODS[priority] || 'info'
        if (typeof this.logger[method] === 'function') {
          this.logger[method](`[C++] ${message}`)
        }
      })
      const verbosity = (this._config && Number.isInteger(this._config.verbosity))
        ? this._config.verbosity
        : DEFAULT_NATIVE_VERBOSITY
      try { binding.setVerbosity(verbosity) } catch (_) {}
      this._nativeLoggerActive = true
    } catch (err) {
      this.logger.warn('Failed to connect native logger:', err && err.message)
    }
  }

  _releaseNativeLogger () {
    if (!this._nativeLoggerActive) return
    try { binding.releaseLogger() } catch (_) {}
    this._nativeLoggerActive = false
  }

  // Framework output callback — same event flow as vla-ggml: a Float32Array
  // output (frames × nCoeffs) lands first, then the RuntimeStats object ends
  // the job; errors arrive with an eventTypeName containing "Error".
  _onAddonEvent (_jsHandle, eventTypeName, outputData, errorData) {
    if (typeof eventTypeName === 'string' && eventTypeName.includes('Error')) {
      const err = new QvacErrorAddonLipsync({
        code: ERR_CODES.INFERENCE_FAILED,
        adds: typeof errorData === 'string' ? errorData : 'native error'
      })
      if (this._pending) this._pending.frames = null
      this._pending = null
      if (this._job.active) this._job.fail(err)
      return
    }
    if (outputData instanceof Float32Array) {
      if (this._pending) this._pending.frames = outputData
      this._job.output(outputData)
      return
    }
    if (outputData && typeof outputData === 'object') {
      const stats = outputData
      const frames = this._pending ? this._pending.frames : null
      this._pending = null
      this._job.end(this.opts.stats ? stats : null, { frames, stats })
    }
  }

  async load ({ backend = 'auto' } = {}) {
    if (backend !== 'auto' && backend !== 'cpu') {
      throw new QvacErrorAddonLipsync({ code: ERR_CODES.INVALID_CONFIG, adds: `backend must be 'auto' or 'cpu' (got: ${backend})` })
    }
    return this._run(async () => {
      if (this.state.configLoaded) return
      await this._load(backend)
      this.state.configLoaded = true
      this.state.weightsLoaded = true
    })
  }

  async _load (backend) {
    this.logger.info('Starting model load')
    this._connectNativeLogger()
    const ggufPath = this._files[0]
    if (!fs.existsSync(ggufPath)) {
      this._releaseNativeLogger()
      throw new QvacErrorAddonLipsync({ code: ERR_CODES.MODEL_NOT_FOUND, adds: ggufPath })
    }
    try {
      const backendsDir = (this._config && this._config.backendsDir)
        ? this._config.backendsDir
        : path.join(__dirname, 'prebuilds')
      this._handle = binding.createInstance(
        this,
        { ggufPath, backend, backendsDir },
        (jsHandle, eventTypeName, outputData, errorData) => {
          this._onAddonEvent(jsHandle, eventTypeName, outputData, errorData)
        }
      )
      binding.activate(this._handle)
      this._hparams = binding.getLipsyncHparams(this._handle)
      this._backendName = binding.getLipsyncBackendName(this._handle)
    } catch (loadError) {
      this.logger.error('Error during model load:', loadError)
      if (this._handle) {
        try { binding.destroyInstance(this._handle) } catch (_) {}
        this._handle = null
      }
      this._releaseNativeLogger()
      throw new QvacErrorAddonLipsync({ code: ERR_CODES.FAILED_TO_LOAD_WEIGHTS, adds: loadError.message, cause: loadError })
    }
    this.logger.info('Model load completed successfully')
  }

  get hparams () { return this._hparams }

  get backendName () { return this._backendName }

  /**
   * Run one inference over a PCM buffer.
   * Resolves with `{ frames, frameCount, fps, stats }` where `frames` is a
   * Float32Array of frameCount × nCoeffs sigmoid coefficients (frame-major,
   * ARKit-52 order).
   */
  async run (input) {
    return this._run(() => this._runInternal(input))
  }

  async _runInternal (input) {
    if (!this._handle) {
      throw new QvacErrorAddonLipsync({ code: ERR_CODES.INSTANCE_NOT_INITIALIZED })
    }
    if (this._hasActiveResponse) {
      throw new QvacErrorAddonLipsync({ code: ERR_CODES.JOB_ALREADY_RUNNING })
    }

    const { idIdx } = validateRunInput(input, this._hparams)

    const response = this._job.start()
    this._pending = { frames: null }

    let accepted = false
    try {
      accepted = binding.runJob(this._handle, {
        type: 'lipsync',
        input: { pcm: input.pcm, idIdx }
      })
    } catch (err) {
      this._pending = null
      this._job.fail(err)
      throw err
    }

    if (!accepted) {
      this._pending = null
      const err = new QvacErrorAddonLipsync({ code: ERR_CODES.JOB_ALREADY_RUNNING })
      this._job.fail(err)
      throw err
    }

    // Clear the busy flag via .finally() on the response promise, not from
    // the native event callback (see vla-ggml for the failure-mode rationale).
    this._hasActiveResponse = true
    const finalized = response.await().finally(() => {
      this._hasActiveResponse = false
    })
    finalized.catch((err) => {
      this.logger?.warn?.('Inference response rejected:', err?.message || err)
    })
    response.await = () => finalized

    const { frames, stats } = await response.await()
    const nCoeffs = this._hparams?.nCoeffs ?? 52
    return {
      frames,
      frameCount: frames ? frames.length / nCoeffs : 0,
      fps: this._hparams?.fps ?? FPS,
      stats
    }
  }

  /**
   * Streaming inference: consume an async iterable of Float32Array PCM
   * chunks (16 kHz mono) and yield `{ frames, frameCount, startTimeMs }`
   * per chunk. Implements the upstream LAM windowing policy: each chunk is
   * right-aligned into a rolling 64-frame window and only the new frames
   * are emitted. Chunks larger than the window are split.
   */
  async * runStreaming (pcmStream, { idIdx = 0 } = {}) {
    const win = new StreamingWindow()
    const nCoeffs = this._hparams?.nCoeffs ?? 52
    let framesEmitted = 0
    for await (const chunk of pcmStream) {
      if (!(chunk instanceof Float32Array)) {
        throw new QvacErrorAddonLipsync({ code: ERR_CODES.INVALID_INPUT, adds: 'pcmStream must yield Float32Array chunks' })
      }
      for (let off = 0; off < chunk.length; off += WINDOW_SAMPLES) {
        const piece = chunk.subarray(off, Math.min(off + WINDOW_SAMPLES, chunk.length))
        const { window, emitFromFrame } = win.push(piece)
        const { frames } = await this.run({ pcm: window, idIdx })
        const out = frames.subarray(emitFromFrame * nCoeffs)
        const outCount = out.length / nCoeffs
        yield {
          frames: out,
          frameCount: outCount,
          startTimeMs: framesEmitted / FPS * 1000
        }
        framesEmitted += outCount
      }
    }
  }

  async pause () { /* no-op: single-shot graph has no per-step cancel point */ }

  async cancel () {
    if (this._handle) {
      try { await binding.cancel(this._handle) } catch (_) {}
    }
  }

  async unload () {
    return this._run(async () => {
      await this.cancel()
      if (this._job.active) {
        this._job.fail(new QvacErrorAddonLipsync({ code: ERR_CODES.MODEL_UNLOADED }))
      }
      this._pending = null
      this._hasActiveResponse = false
      if (this._handle) {
        try {
          binding.destroyInstance(this._handle)
        } catch (destroyError) {
          this._handle = null
          this._releaseNativeLogger()
          throw new QvacErrorAddonLipsync({ code: ERR_CODES.FAILED_TO_DESTROY, adds: destroyError.message, cause: destroyError })
        }
        this._handle = null
      }
      this._releaseNativeLogger()
      this._hparams = null
      this._backendName = null
      this.state.configLoaded = false
      this.state.weightsLoaded = false
    })
  }

  getState () { return this.state }
}

module.exports = LipsyncModel
module.exports.LipsyncModel = LipsyncModel
module.exports.SAMPLE_RATE = SAMPLE_RATE
module.exports.FPS = FPS
module.exports.WINDOW_FRAMES = WINDOW_FRAMES
module.exports.WINDOW_SAMPLES = WINDOW_SAMPLES
module.exports.framesForSamples = framesForSamples
module.exports.int16ToFloat32 = int16ToFloat32
module.exports.QvacErrorAddonLipsync = QvacErrorAddonLipsync
module.exports.ERR_CODES = ERR_CODES
