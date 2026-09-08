import assert from 'node:assert/strict'
import { readFileSync, writeFileSync } from 'node:fs'
import { createHash } from 'node:crypto'
import { createRequire } from 'node:module'
import path from 'node:path'

const require = createRequire(import.meta.url)
const phase = process.argv[2]
const sdkRoot = path.dirname(require.resolve('@qvac/sdk/package.json'))
const sourceCommit = '5ae0616cdae0c17fc7ac7a007a85e8b78269c13e'
if (phase === 'patch') {
  const response = await fetch(`https://raw.githubusercontent.com/localhost41/qvac/${sourceCommit}/packages/sdk/src/client/rpc/worker-startup-error.ts`)
  assert.equal(response.status, 200)
  const source = await response.text()
  writeFileSync('patch-source.ts', source)
  const ts = require('typescript')
  const output = ts.transpileModule(source.replace("'@/utils/errors-client'", "'../../utils/errors-client.js'"), {
    compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ES2022 }
  }).outputText
  writeFileSync(path.join(sdkRoot, 'dist/src/client/rpc/worker-startup-error.js'), output)
  writeFileSync('patch-result.json', JSON.stringify({ sourceCommit, sourceSha256: createHash('sha256').update(source).digest('hex'), changedInstalledFile: 'dist/src/client/rpc/worker-startup-error.js' }, null, 2))
  process.exit(0)
}
assert(['baseline', 'patched', 'fixed'].includes(phase))
const sdk = await import('@qvac/sdk')
let caught
try {
  await sdk.heartbeat()
} catch (error) {
  caught = error
} finally {
  await sdk.close()
}
function serialize(error) {
  if (!error) return null
  return Object.fromEntries(['name', 'message', 'code', 'workerExited', 'exitCode', 'exitSignal', 'stderrTail'].filter(key => error[key] !== undefined).map(key => [key, error[key]]).concat(error.cause ? [['cause', serialize(error.cause)]] : []))
}
const result = { phase, sdk: require('@qvac/sdk/package.json').version, platform: process.platform, arch: process.arch, node: process.version, heartbeat: caught ? 'failed' : 'passed', error: serialize(caught) }
writeFileSync(`${phase}-result.json`, JSON.stringify(result, null, 2))
console.log(JSON.stringify(result, null, 2))
assert.equal(result.sdk, '0.19.0')
if (phase === 'fixed') {
  assert.equal(caught, undefined, 'worker must start after installing libatomic1')
} else {
  assert(caught?.cause instanceof sdk.WorkerStartupError, 'startup cause must retain its SDK type')
  assert.match(caught.cause.stderrTail, /libatomic\.so\.1: cannot open shared object file: No such file or directory/)
  assert.equal(caught.cause.workerExited, true)
  if (phase === 'baseline') {
    assert.doesNotMatch(caught.cause.message, /install libatomic1/)
  } else {
    assert.match(caught.cause.message, /install libatomic1/)
    const baseline = JSON.parse(readFileSync('baseline-result.json', 'utf8')).error.cause
    for (const key of ['code', 'exitCode', 'exitSignal', 'workerExited', 'stderrTail']) assert.equal(caught.cause[key], baseline[key], `preserve ${key}`)
  }
}
