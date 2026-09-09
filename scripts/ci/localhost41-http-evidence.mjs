import assert from 'node:assert/strict'
import { spawn } from 'node:child_process'
import { readFileSync, writeFileSync } from 'node:fs'
import { createRequire } from 'node:module'
import path from 'node:path'
import { setTimeout as sleep } from 'node:timers/promises'
const phase = process.argv[2]
const artifact = path.resolve('../artifacts')
const require = createRequire(path.resolve('package.json'))
const sdkEntry = require.resolve('@qvac/sdk')
const cliEntry = require.resolve('@qvac/cli')
const cliRequire = createRequire(cliEntry)
assert.equal(cliRequire.resolve('@qvac/sdk'), sdkEntry, 'CLI must consume exactly the selected SDK package')
const config = {
  rpcInitTimeoutMs: 30000,
  cacheDirectory: '/tmp/qvac-http-evidence-cache',
  serve: { load: { lazy: true, timeoutMs: null }, models: { 'startup-probe': { model: 'QWEN3_600M_INST_Q4', preload: false } } }
}
writeFileSync('qvac.config.json', JSON.stringify(config, null, 2))
const logs = []
const began = performance.now()
const child = spawn(process.execPath, [cliEntry, 'serve', '--host', '127.0.0.1', '--port', '32187'], { cwd: process.cwd(), env: process.env, stdio: ['ignore','pipe','pipe'] })
for (const pipe of [child.stdout, child.stderr]) pipe.on('data', c => logs.push({ms: performance.now()-began,text:c.toString()}))
let result
try {
  let ready = false
  for (let i=0;i<240;i++) {
    if (child.exitCode !== null) throw new Error(`serve exited ${child.exitCode}`)
    try { const res=await fetch('http://127.0.0.1:32187/v1/models'); if(res.ok) { ready=true;break } } catch {}
    await sleep(250)
  }
  assert(ready, 'serve must listen within 60 seconds')
  const listeningMs=performance.now()-began
  const requestAt=performance.now()
  const res=await fetch('http://127.0.0.1:32187/v1/chat/completions', { method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({model:'startup-probe',messages:[{role:'user',content:'hi'}]}) })
  const body=await res.json()
  result={phase, sdkEntry, cliEntry, listeningMs, requestMs:performance.now()-requestAt, status:res.status, body}
  writeFileSync(path.join(artifact,`${phase}-http.json`),JSON.stringify(result,null,2))
  assert.equal(res.status,503)
  assert.equal(body.error.code,'model_load_failed')
  assert.match(body.error.message,/RPC initialization timed out after 30000ms/)
  assert.doesNotMatch(body.error.message,/install libatomic1/)
} finally {
  child.kill('SIGTERM')
  await Promise.race([new Promise(resolve=>child.once('exit',resolve)),sleep(5000)])
  if(child.exitCode===null) child.kill('SIGKILL')
  writeFileSync(path.join(artifact,`${phase}-serve-log.json`),JSON.stringify(logs,null,2))
  writeFileSync(path.join(artifact,`${phase}-serve.log`),logs.map(x=>x.text).join(''))
}
// Fresh process above exercises the real HTTP route. This process captures the
// actual public SDK error as a separate control without substituting a fixture.
const sdk=await import(sdkEntry)
let error
try { await sdk.heartbeat() } catch(e) {error=e} finally { await sdk.close() }
assert(error?.cause instanceof sdk.WorkerStartupError)
assert.match(error.cause.stderrTail,/libatomic\.so\.1: cannot open shared object file: No such file or directory/)
assert.equal(error.cause.workerExited,true)
assert.equal(error.cause.exitCode,null)
assert.equal(error.cause.exitSignal,'SIGABRT')
if(phase==='branch') assert.match(error.cause.message,/install libatomic1/)
else assert.doesNotMatch(error.cause.message,/install libatomic1/)
const serialize=e=>e ? Object.fromEntries(['name','message','code','workerExited','exitCode','exitSignal','stderrTail'].filter(k=>e[k]!==undefined).map(k=>[k,e[k]]).concat(e.cause ? [['cause',serialize(e.cause)]]:[])) : null
writeFileSync(path.join(artifact,`${phase}-sdk-error.json`),JSON.stringify(serialize(error),null,2))
console.log(JSON.stringify(result,null,2))
