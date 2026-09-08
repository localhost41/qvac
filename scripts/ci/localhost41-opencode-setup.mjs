import assert from 'node:assert/strict'
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs'
import { createRequire } from 'node:module'
import { execFileSync } from 'node:child_process'
import path from 'node:path'
import { pathToFileURL } from 'node:url'

const releaseCommit = '80c0973ea5c6ead90c7f1d6985adbba90755c7d0'
const verifierCommit = '5877dc5318d5ac9a0625b1606189512d1da76e5e'
if (process.argv[2] === 'verify') {
  const versions = {}
  for (const name of ['opencode-ai', '@qvac/opencode-plugin', '@qvac/ai-sdk-provider', '@qvac/cli', '@qvac/sdk']) {
    const pkg = JSON.parse(readFileSync(path.join('node_modules', name, 'package.json'), 'utf8'))
    versions[name] = pkg.version
  }
  assert.equal(versions['@qvac/sdk'], '0.19.0')
  assert.equal(versions['@qvac/cli'], '0.13.0')
  assert.equal(versions['@qvac/ai-sdk-provider'], '0.7.0')
  assert.equal(versions['@qvac/opencode-plugin'], '0.3.1')
  for (const name of ['@qvac/opencode-plugin', '@qvac/ai-sdk-provider', '@qvac/cli']) {
    const localRequire = createRequire(path.resolve('node_modules', name, 'package.json'))
    const sdkPath = localRequire.resolve('@qvac/sdk/package.json')
    assert.equal(JSON.parse(readFileSync(sdkPath, 'utf8')).version, '0.19.0')
  }
  writeFileSync('../artifacts/versions.json', JSON.stringify({ versions, releaseCommit, verifierCommit, note: 'Plugin 0.3.0 published dist with exact 0.3.1 release-PR package.json; 0.3.1 is not yet published. All other packages are published versions.' }, null, 2))
  const config = JSON.parse(readFileSync('opencode.json', 'utf8'))
  config.plugin[0][0] = pathToFileURL(path.resolve('candidate-plugin/dist/index.js')).href
  writeFileSync('opencode.json', JSON.stringify(config, null, 2))
  writeFileSync('../artifacts/opencode.json', readFileSync('opencode.json'))
  console.log(versions)
  process.exit(0)
}
async function get(url) {
  const response = await fetch(url)
  assert.equal(response.status, 200, url)
  return await response.text()
}
const packed = JSON.parse(execFileSync('npm', ['pack', '@qvac/opencode-plugin@0.3.0', '--json'], { encoding: 'utf8' }))[0]
mkdirSync('candidate-plugin', { recursive: true })
execFileSync('tar', ['-xzf', packed.filename, '-C', 'candidate-plugin', '--strip-components=1'])
const manifest = await get(`https://raw.githubusercontent.com/tetherto/qvac/${releaseCommit}/plugins/opencode/package.json`)
const release = JSON.parse(manifest)
assert.equal(release.version, '0.3.1')
writeFileSync('candidate-plugin/package.json', manifest)
writeFileSync('package.json', JSON.stringify({ name: 'qvac019-opencode-runtime-evidence', private: true, dependencies: { 'opencode-ai': '1.18.29', '@qvac/opencode-plugin': 'file:./candidate-plugin', '@qvac/ai-sdk-provider': '0.7.0', '@qvac/cli': '0.13.0', '@qvac/sdk': '0.19.0' } }, null, 2))
writeFileSync('opencode.json', JSON.stringify({ $schema: 'https://opencode.ai/config.json', plugin: [['@qvac/opencode-plugin', { model: 'qwen3.5-0.8b', ctxSize: 32768, reasoningBudget: 0, tools: true }]] }, null, 2))
writeFileSync('../artifacts/verify-opencode-run-output.cjs', await get(`https://raw.githubusercontent.com/localhost41/qvac/${verifierCommit}/scripts/ci/verify-opencode-run-output.cjs`))
writeFileSync('../artifacts/opencode.json', readFileSync('opencode.json'))
writeFileSync('../artifacts/plugin-release-package.json', manifest)
