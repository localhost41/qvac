import {
  loadModel,
  textToSpeech,
  unloadModel
} from '@qvac/sdk'
import { createWav, playAudio, int16ArrayToBuffer, createWavHeader } from './utils'

// CosyVoice3 TTS (GGML): native, CPU, on-device text -> 24 kHz speech.
//
// CosyVoice3 loads a *directory* of GGUFs (LM/flow/HiFT) + a baked voice.gguf +
// the Qwen2 BPE tokenizer (vocab.json/merges.txt), assembled by
// qvac-ext-lib-whisper.cpp/tts-cpp/scripts/assemble-cosyvoice3-model.py.  There
// is no registry entry yet, so this demo points `modelSrc` at the LM GGUF
// *inside* that directory — the tts-ggml plugin derives the model dir from its
// parent (dirname).  A future registry constant (e.g. TTS_COSYVOICE3_0_5B) would
// drop in place of the local path, exactly like the supertonic example.
const COSYVOICE_SAMPLE_RATE = 24000

const modelDir = String(process.env['COSYVOICE_MODEL_DIR'] ?? '')
if (!modelDir) {
  console.error('✖ Set COSYVOICE_MODEL_DIR to an assembled CosyVoice3 model directory.')
  process.exit(1)
}
// The primary modelSrc is the LLM GGUF; the plugin discovers flow/HiFT/voice/
// tokenizer next to it in the same directory. A future registry constant would
// drop in place of this local path (with modelType inferred), like supertonic.
const modelSrc = `${modelDir}/cosyvoice3-llm-f32.gguf`

try {
  const modelId = await loadModel({
    modelType: 'tts-ggml',
    modelSrc,
    modelConfig: {
      ttsEngine: 'cosyvoice3',
      language: 'en'
    }
  })

  console.log(`▸ Model loaded: ${modelId}`)

  console.log('▸ Testing Text-to-Speech...')
  const result = textToSpeech({
    modelId,
    text: 'Peer to peer, local first, and fully on device. No cloud required.',
    inputType: 'text',
    stream: false
  })

  const audioBuffer = await result.buffer
  console.log(`▸ TTS complete. Total samples: ${audioBuffer.length}`)

  console.log('▸ Saving audio to file...')
  createWav(audioBuffer, COSYVOICE_SAMPLE_RATE, 'cosyvoice-output.wav')
  console.log('▸ Audio saved to cosyvoice-output.wav')

  console.log('▸ Playing audio...')
  const audioData = int16ArrayToBuffer(audioBuffer)
  const wavBuffer = Buffer.concat([
    createWavHeader(audioData.length, COSYVOICE_SAMPLE_RATE),
    audioData
  ])
  playAudio(wavBuffer)
  console.log('▸ Audio playback complete')

  await unloadModel({ modelId })
  console.log('▸ Model unloaded')
  process.exit(0)
} catch (error) {
  console.error('✖', error)
  process.exit(1)
}
