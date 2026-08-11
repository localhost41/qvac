import test from 'brittle'
import { existsSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { MOBILE_HOSTS, runIOSAddonLinker } from '@/expo/plugins/withMobileBundle'

test('MOBILE_HOSTS: canonical mobile host set', (t) => {
  t.alike(MOBILE_HOSTS, ['android-arm64', 'ios-arm64', 'ios-arm64-simulator', 'ios-x64-simulator'])
})

test('runIOSAddonLinker: waits for the linker to finish', async (t) => {
  const fixtureDir = mkdtempSync(join(tmpdir(), 'qvac-ios-linker-'))
  const linkerPath = join(fixtureDir, 'link.mjs')
  const markerPath = join(fixtureDir, 'linked')
  t.teardown(() => rmSync(fixtureDir, { recursive: true, force: true }))

  writeFileSync(
    linkerPath,
    `import { writeFileSync } from 'node:fs'\nwriteFileSync(${JSON.stringify(markerPath)}, '')\n`
  )

  await runIOSAddonLinker(linkerPath)

  t.ok(existsSync(markerPath), 'linker completed before the helper resolved')
})

test('runIOSAddonLinker: rejects when the linker fails', async (t) => {
  const fixtureDir = mkdtempSync(join(tmpdir(), 'qvac-ios-linker-'))
  const linkerPath = join(fixtureDir, 'link.mjs')
  t.teardown(() => rmSync(fixtureDir, { recursive: true, force: true }))

  writeFileSync(linkerPath, `console.error('fixture failure')\nprocess.exit(7)\n`)

  let error: Error | undefined
  try {
    await runIOSAddonLinker(linkerPath)
  } catch (cause) {
    error = cause as Error
  }

  t.ok(error, 'expected the helper to reject')
  t.ok(error?.message.includes('code 7'), 'reports the child exit code')
  t.ok(error?.message.includes('fixture failure'), 'reports linker diagnostics')
})
