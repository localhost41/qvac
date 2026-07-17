'use strict'

// Parler engine integration smoke + description/emotion surface coverage.
// Mirrors supertonic.test.js but adds the parler-specific description
// template matrix (constructor vs per-call, conflicts, emotion switching)
// and an indic-variant leg with a Hindi prompt.

const os = require('bare-os')
const path = require('bare-path')
const test = require('brittle')

const { loadParlerTTS, runParlerTTS } = require('../utils/runParlerTTS')
const { ensureParlerModel } = require('../utils/downloadModel')
const { recordTtsStats } = require('../utils/perf-helper')

const platform = os.platform()
const isMobile = platform === 'ios' || platform === 'android'

function getBaseDir() {
  return isMobile && global.testDir ? global.testDir : '.'
}

const MODEL_MISSING =
  'Parler GGUF not available - registry fetch failed. Run `npm run download-models:registry -- --group parler` or stage models locally.'

test(
  'Parler TTS (ggml): all-defaults synthesis returns 44.1 kHz audio + stats',
  { timeout: 600000 },
  async (t) => {
    const baseDir = getBaseDir()
    const download = await ensureParlerModel({ targetDir: path.join(baseDir, 'models') })
    if (!download.success) {
      t.fail(MODEL_MISSING)
      return
    }

    // Deliberately NO description/voice/emotion: the engine renders the
    // models' recommended fallback caption — everything works on defaults.
    const model = await loadParlerTTS({ parlerModelPath: download.path, seed: 42 })
    try {
      const wavPath = isMobile ? undefined : path.join(baseDir, 'test', 'output', 'parler-en.wav')
      const text = 'The parler engine speaks with a voice controlled by a text description.'
      const t0 = Date.now()
      const result = await runParlerTTS(
        model,
        { text, saveWav: !isMobile, wavOutputPath: wavPath },
        { minSamples: 10000, maxSamples: 5000000, minDurationMs: 250, maxDurationMs: 300000 }
      )
      const wallMs = Date.now() - t0
      console.log(result.output)

      t.ok(result.passed, 'parler synth passes expectations')
      t.ok(result.data.sampleCount > 0, 'parler produced audio')
      t.is(result.data.reportedSampleRate, 44100, 'parler reports 44.1 kHz native sample rate')

      const st = result.data?.stats || {}
      t.comment(
        recordTtsStats(
          'parler en',
          {
            realTimeFactor: st.realTimeFactor,
            audioDurationMs: st.audioDurationMs || result.data?.durationMs,
            totalSamples: st.totalSamples,
            backendDevice: st.backendDevice
          },
          { wallMs, sampleCount: result.data?.sampleCount, model: 'parler', output: text }
        )
      )
      if (result.data.stats) {
        t.ok(typeof result.data.stats.realTimeFactor === 'number', 'parler stats include RTF')
        t.ok(
          typeof result.data.stats.audioDurationMs === 'number',
          'parler stats include audio duration'
        )
        t.ok(
          typeof result.data.stats.backendDevice === 'number',
          'parler stats include backendDevice'
        )
        t.ok(typeof result.data.stats.backendId === 'number', 'parler stats include backendId')
      }
    } finally {
      try {
        await model.unload()
      } catch (_e) {}
    }
  }
)

test(
  'Parler TTS (ggml): cancel mid-flight rejects the response',
  { timeout: 600000 },
  async (t) => {
    const baseDir = getBaseDir()
    const download = await ensureParlerModel({ targetDir: path.join(baseDir, 'models') })
    if (!download.success) {
      t.fail(MODEL_MISSING)
      return
    }

    const model = await loadParlerTTS({ parlerModelPath: download.path, voice: 'Laura' })
    try {
      const response = await model.run({
        type: 'text',
        input: 'Cancel this synthesis call before it completes.'
      })
      setTimeout(() => {
        response.cancel().catch(() => {})
      }, 50)

      let failed = false
      try {
        await response.await()
      } catch (e) {
        failed = true
        console.log('  cancel rejected with: ' + e.message)
      }
      t.ok(failed, 'cancelled parler response should reject')
    } finally {
      try {
        await model.unload()
      } catch (_e) {}
    }
  }
)

test(
  'Parler TTS (ggml): description matrix — conflicts throw, per-call wins, defaults work',
  { timeout: 600000 },
  async (t) => {
    const baseDir = getBaseDir()
    const download = await ensureParlerModel({ targetDir: path.join(baseDir, 'models') })
    if (!download.success) {
      t.fail(MODEL_MISSING)
      return
    }

    const TTSGgml = require('@qvac/tts-ggml')

    // Same-level conflict at the constructor throws synchronously.
    t.exception(
      () =>
        new TTSGgml({
          engine: TTSGgml.ENGINE_PARLER,
          files: { parlerModel: download.path },
          description: 'A calm female voice.',
          emotion: 'happy'
        }),
      /mutually exclusive/,
      'constructor description + emotion throws'
    )

    // Wrong-engine guard: parler-only options on supertonic throw.
    t.exception(
      () =>
        new TTSGgml({
          engine: TTSGgml.ENGINE_SUPERTONIC,
          files: { supertonicModel: download.path },
          emotion: 'happy'
        }),
      /parler-only/,
      'emotion on a supertonic instance throws'
    )

    const model = await loadParlerTTS({
      parlerModelPath: download.path,
      voice: 'Laura',
      seed: 42
    })
    try {
      // Per-call description conflict (same level) rejects.
      await t.exception(
        model.run({
          input: 'Conflict check.',
          description: 'A calm voice.',
          emotion: 'sad'
        }),
        /mutually exclusive/,
        'per-call description + template field rejects'
      )

      // Constructor template + per-call template merge runs fine.
      const merged = await runParlerTTS(
        model,
        { text: 'Merged template synthesis.', perCall: { emotion: 'happy' } },
        { minSamples: 5000 }
      )
      t.ok(merged.passed, 'ctor voice + per-call emotion merges and synthesizes')

      // Per-call full description wins for that call.
      const overridden = await runParlerTTS(
        model,
        {
          text: 'Description override synthesis.',
          perCall: { description: 'A deep male voice speaks slowly, very clear audio.' }
        },
        { minSamples: 5000 }
      )
      t.ok(overridden.passed, 'per-call description overrides the constructor template')
    } finally {
      try {
        await model.unload()
      } catch (_e) {}
    }

    // Constructor-level free-text description + per-call template rejects.
    const descModel = await loadParlerTTS({
      parlerModelPath: download.path,
      description: 'A calm female voice, very clear audio.',
      seed: 42
    })
    try {
      await t.exception(
        descModel.run({ input: 'Cross-level conflict.', emotion: 'sad' }),
        /constructor-level description/,
        'per-call template + ctor description rejects'
      )
    } finally {
      try {
        await descModel.unload()
      } catch (_e) {}
    }
  }
)

test(
  'Parler TTS (ggml): emotion flag changes the audio; per-call switch needs no reload',
  { timeout: 600000 },
  async (t) => {
    const baseDir = getBaseDir()
    const download = await ensureParlerModel({ targetDir: path.join(baseDir, 'models') })
    if (!download.success) {
      t.fail(MODEL_MISSING)
      return
    }

    const model = await loadParlerTTS({
      parlerModelPath: download.path,
      voice: 'Laura',
      seed: 42
    })
    try {
      const text = 'The weather is wonderful today.'
      const happy = await runParlerTTS(model, { text, perCall: { emotion: 'happy' } })
      const sad = await runParlerTTS(model, { text, perCall: { emotion: 'sad' } })
      t.ok(happy.passed && sad.passed, 'both emotion runs synthesize')
      const identical =
        happy.data.sampleCount === sad.data.sampleCount &&
        happy.data.samples.every((v, i) => v === sad.data.samples[i])
      t.ok(!identical, 'happy vs sad conditioning produces different audio (same seed/text)')

      // Invalid emotion is rejected with the valid list (validated natively,
      // so the failure surfaces on the response, not on run() itself).
      const bad = await model.run({ input: text, emotion: 'angry' })
      let badMsg = ''
      try {
        await bad.await()
      } catch (e) {
        badMsg = String((e && e.message) || e)
      }
      t.ok(/valid:/.test(badMsg), `invalid emotion rejects listing valid values (got: ${badMsg})`)
    } finally {
      try {
        await model.unload()
      } catch (_e) {}
    }
  }
)

test(
  'Parler TTS (ggml): fixed seed is deterministic across runs and reload',
  { timeout: 600000 },
  async (t) => {
    const baseDir = getBaseDir()
    const download = await ensureParlerModel({ targetDir: path.join(baseDir, 'models') })
    if (!download.success) {
      t.fail(MODEL_MISSING)
      return
    }

    const model = await loadParlerTTS({
      parlerModelPath: download.path,
      voice: 'Laura',
      emotion: 'neutral',
      seed: 42
    })
    try {
      const text = 'Deterministic synthesis check.'
      const a = await runParlerTTS(model, { text })
      // The engine reseeds from EngineOptions::seed on every synthesize
      // call, so a same-instance repeat must be bit-identical.
      const b = await runParlerTTS(model, { text })
      t.is(a.data.sampleCount, b.data.sampleCount, 'repeat run has identical length')
      t.ok(
        a.data.samples.every((v, i) => v === b.data.samples[i]),
        'repeat run is bit-identical (fixed seed)'
      )

      await model.reload({ emotion: 'neutral' })
      const c = await runParlerTTS(model, { text })
      t.is(a.data.sampleCount, c.data.sampleCount, 'post-reload run has identical length')
      t.ok(
        a.data.samples.every((v, i) => v === c.data.samples[i]),
        'post-reload run is bit-identical'
      )
    } finally {
      try {
        await model.unload()
      } catch (_e) {}
    }
  }
)

test(
  'Parler TTS (ggml): runStream emits per-sentence chunks with one pinned description',
  { timeout: 600000 },
  async (t) => {
    const baseDir = getBaseDir()
    const download = await ensureParlerModel({ targetDir: path.join(baseDir, 'models') })
    if (!download.success) {
      t.fail(MODEL_MISSING)
      return
    }

    const TTSGgml = require('@qvac/tts-ggml')
    const model = new TTSGgml({
      engine: TTSGgml.ENGINE_PARLER,
      files: { parlerModel: download.path },
      seed: 42,
      opts: { stats: true }
    })
    await model.load()
    try {
      const text = 'First sentence. Second sentence here.'
      const response = await model.runStream(text, {
        maxChunkScalars: 20,
        voice: 'Laura',
        emotion: 'happy'
      })
      const updates = []
      await response
        .onUpdate((d) => {
          if (d && d.outputArray) updates.push(d)
        })
        .await()

      t.ok(updates.length >= 2, `runStream produced multiple chunks (got ${updates.length})`)
      t.is(updates.filter((u) => u.isLast).length, 1, 'exactly one isLast=true emitted')
      t.ok(
        updates.every((u) => u.sampleRate === 44100),
        'every chunk reports 44.1 kHz'
      )
    } finally {
      try {
        await model.unload()
      } catch (_e) {}
    }
  }
)

test(
  'Parler TTS (ggml): indic variant synthesizes a Hindi prompt with an emotion flag',
  { timeout: 900000 },
  async (t) => {
    const baseDir = getBaseDir()
    const download = await ensureParlerModel({
      targetDir: path.join(baseDir, 'models'),
      variant: 'indic'
    })
    if (!download.success) {
      t.fail(MODEL_MISSING)
      return
    }

    const model = await loadParlerTTS({
      parlerModelPath: download.path,
      voice: 'Rohit',
      emotion: 'happy',
      seed: 42
    })
    try {
      const wavPath = isMobile ? undefined : path.join(baseDir, 'test', 'output', 'parler-hi.wav')
      const text = 'आज मौसम बहुत अच्छा है और हम सब बाहर घूमने जा रहे हैं।'
      const result = await runParlerTTS(
        model,
        { text, saveWav: !isMobile, wavOutputPath: wavPath },
        { minSamples: 10000, minDurationMs: 250, maxDurationMs: 300000 }
      )
      console.log(result.output)
      t.ok(result.passed, 'indic hindi synth passes expectations')
      t.is(result.data.reportedSampleRate, 44100, 'indic reports 44.1 kHz')
    } finally {
      try {
        await model.unload()
      } catch (_e) {}
    }
  }
)
