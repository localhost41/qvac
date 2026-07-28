import { fork, spawn } from 'node:child_process'
import fs from 'node:fs'
import { fileURLToPath, pathToFileURL } from 'node:url'
import { resolveSdkEntrypoint } from './checks/project.js'
import {
  isDeepProbeMessage,
  isDeepProbeProtocolCandidate,
  type DeepProbeMessage,
  type DeepProbePhase,
  type SerializedProbeError
} from './deep-protocol.js'
import type { CheckResult, CheckSection } from './types.js'

const DEFAULT_TIMEOUT_MS = 45_000
const MAX_OUTPUT_CHARS = 16_384
const TERMINATION_GRACE_MS = 2_000

export interface SdkRuntimeProbeResult {
  outcome: 'pass' | 'fail' | 'timeout' | 'spawn-error' | 'protocol-error'
  durationMs: number
  stdout: string
  stderr: string
  exitCode: number | null
  signal: NodeJS.Signals | null
  phase?: DeepProbePhase | undefined
  probeMessage?: DeepProbeMessage | undefined
  error?: string | undefined
}

export interface SdkRuntimeProbeOptions {
  timeoutMs?: number | undefined
  maxOutputChars?: number | undefined
  nodePath?: string | undefined
  childModulePath?: string | undefined
}

function appendBounded(current: string, chunk: string, limit: number): string {
  const next = current + chunk
  return next.length <= limit ? next : next.slice(next.length - limit)
}

function defaultChildModulePath(): string {
  const compiledPath = fileURLToPath(new URL('./deep-probe-child.js', import.meta.url))
  if (fs.existsSync(compiledPath)) return compiledPath
  return fileURLToPath(new URL('./deep-probe-child.ts', import.meta.url))
}

function spawnErrorResult(startedAt: number, error: unknown): SdkRuntimeProbeResult {
  return {
    outcome: 'spawn-error',
    durationMs: Date.now() - startedAt,
    stdout: '',
    stderr: '',
    exitCode: null,
    signal: null,
    error: error instanceof Error ? error.message : String(error)
  }
}

function signalProbeTree(child: ReturnType<typeof fork>, signal: NodeJS.Signals): void {
  const pid = child.pid
  if (pid === undefined) return

  if (process.platform === 'win32') {
    const killer = spawn('taskkill', ['/PID', String(pid), '/T', '/F'], {
      stdio: 'ignore',
      windowsHide: true
    })
    killer.once('error', () => child.kill(signal))
    return
  }

  try {
    process.kill(-pid, signal)
  } catch (error) {
    const code = (error as NodeJS.ErrnoException).code
    if (code !== 'ESRCH') child.kill(signal)
  }
}

export function probeSdkRuntime(
  entrypoint: string,
  projectRoot: string,
  options: SdkRuntimeProbeOptions = {}
): Promise<SdkRuntimeProbeResult> {
  const timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS
  const maxOutputChars = options.maxOutputChars ?? MAX_OUTPUT_CHARS
  const startedAt = Date.now()
  let child: ReturnType<typeof fork>
  try {
    child = fork(
      options.childModulePath ?? defaultChildModulePath(),
      [pathToFileURL(entrypoint).href],
      {
        cwd: projectRoot,
        detached: process.platform !== 'win32',
        env: { ...process.env, FORCE_COLOR: '0', NO_COLOR: '1' },
        execPath: options.nodePath ?? process.execPath,
        stdio: ['ignore', 'pipe', 'pipe', 'ipc']
      }
    )
  } catch (error) {
    return Promise.resolve(spawnErrorResult(startedAt, error))
  }

  let stdout = ''
  let stderr = ''
  let timedOut = false
  let spawnError: string | undefined
  let protocolMessage: DeepProbeMessage | undefined
  let protocolCandidateCount = 0
  child.stdout?.on('data', (chunk: Buffer) => {
    stdout = appendBounded(stdout, chunk.toString('utf8'), maxOutputChars)
  })
  child.stderr?.on('data', (chunk: Buffer) => {
    stderr = appendBounded(stderr, chunk.toString('utf8'), maxOutputChars)
  })
  child.on('message', (message: unknown) => {
    if (!isDeepProbeProtocolCandidate(message)) return
    protocolCandidateCount += 1
    if (isDeepProbeMessage(message) && protocolMessage === undefined) protocolMessage = message
  })

  return new Promise((resolve) => {
    let terminationTimer: NodeJS.Timeout | undefined
    const timeoutTimer = setTimeout(() => {
      timedOut = true
      signalProbeTree(child, 'SIGTERM')
      terminationTimer = setTimeout(() => signalProbeTree(child, 'SIGKILL'), TERMINATION_GRACE_MS)
      terminationTimer.unref()
    }, timeoutMs)
    timeoutTimer.unref()

    child.once('error', (error) => {
      spawnError = error.message
    })
    child.once('close', (exitCode, signal) => {
      clearTimeout(timeoutTimer)
      if (terminationTimer !== undefined) clearTimeout(terminationTimer)
      if (timedOut && process.platform !== 'win32') signalProbeTree(child, 'SIGKILL')

      let outcome: SdkRuntimeProbeResult['outcome']
      let error = spawnError
      if (spawnError !== undefined) {
        outcome = 'spawn-error'
      } else if (timedOut) {
        outcome = 'timeout'
      } else if (protocolCandidateCount !== 1 || protocolMessage === undefined) {
        outcome = 'protocol-error'
        error =
          protocolCandidateCount === 0
            ? 'Probe exited without a result message.'
            : 'Probe emitted an invalid or duplicate result message.'
      } else if (protocolMessage.ok && exitCode === 0) {
        outcome = 'pass'
      } else if (!protocolMessage.ok && exitCode !== 0) {
        outcome = 'fail'
      } else {
        outcome = 'protocol-error'
        error = 'Probe result message did not agree with its exit code.'
      }

      resolve({
        outcome,
        durationMs: Date.now() - startedAt,
        stdout: stdout.trim(),
        stderr: stderr.trim(),
        exitCode,
        signal,
        ...(protocolMessage !== undefined
          ? { phase: protocolMessage.phase, probeMessage: protocolMessage }
          : {}),
        ...(error !== undefined ? { error } : {})
      })
    })
  })
}

function formatSerializedError(error: SerializedProbeError, label: string): string {
  const attributes = [
    error.code !== undefined ? `code=${String(error.code)}` : '',
    error.exitCode !== undefined ? `exitCode=${String(error.exitCode)}` : '',
    error.exitSignal !== undefined ? `exitSignal=${String(error.exitSignal)}` : ''
  ].filter(Boolean)
  const heading = attributes.length > 0 ? `${label} (${attributes.join(', ')}):` : `${label}:`
  const current = `${heading}\n${error.stack ?? `${error.name}: ${error.message}`}`
  return error.cause === undefined
    ? current
    : `${current}\n${formatSerializedError(error.cause, 'Caused by')}`
}

function formatDiagnostics(result: SdkRuntimeProbeResult): string | undefined {
  const failure = result.probeMessage?.ok === false ? result.probeMessage : undefined
  const parts = [
    result.error ? `Probe error:\n${result.error}` : '',
    failure ? formatSerializedError(failure.error, `Failure during ${failure.phase}`) : '',
    failure?.cleanupError ? formatSerializedError(failure.cleanupError, 'Cleanup failure') : '',
    result.stderr ? `stderr:\n${result.stderr}` : '',
    result.stdout ? `stdout:\n${result.stdout}` : ''
  ].filter(Boolean)
  return parts.length > 0 ? parts.join('\n\n') : undefined
}

function diagnosticText(result: SdkRuntimeProbeResult): string {
  return formatDiagnostics(result) ?? ''
}

export function classifySdkRuntimeFailure(result: SdkRuntimeProbeResult): string {
  const diagnostics = diagnosticText(result)

  if (
    result.signal === 'SIGILL' ||
    /exitSignal=SIGILL|signal SIGILL|illegal instruction|\bSIGILL\b/i.test(diagnostics)
  ) {
    return 'A native addon used an instruction unsupported by this CPU. Use a compatible addon build or a worker bundle that excludes the affected plugin.'
  }
  if (/GLIBCXX_[\d.]+.*not found|version [`\'"]GLIBCXX_/i.test(diagnostics)) {
    return 'The host libstdc++ is older than a native addon requires. Update libstdc++ or use an addon build compatible with this distribution.'
  }
  if (/VCRUNTIME\d*\.dll|MSVCP\d*\.dll|Visual C\+\+.*Redistributable/i.test(diagnostics)) {
    return 'A Microsoft Visual C++ runtime dependency is missing. Install the current Visual C++ Redistributable and retry.'
  }
  if (
    /VK_ERROR_|vkCreateInstance|vkEnumerateInstance|(?:lib)?vulkan[^\n]*(failed|error|not found|unsupported|version|cannot open)/i.test(
      diagnostics
    )
  ) {
    return 'A Vulkan dependency failed to load or initialize. Install or update the Vulkan loader and GPU driver to versions providing Vulkan 1.4 or newer.'
  }
  if (
    /error while loading shared libraries|cannot open shared object file|Library not loaded|The specified module could not be found/i.test(
      diagnostics
    )
  ) {
    return 'A native shared-library dependency could not be loaded. Re-run with --verbose to identify the missing library.'
  }
  if (
    /BARE_RUNTIME_BINARY_NOT_FOUND|BareRuntimeBinaryNotFoundError|Bare runtime binary.*not found/i.test(
      diagnostics
    )
  ) {
    return 'The Bare runtime binary is missing. Reinstall @qvac/sdk with lifecycle scripts enabled for this host.'
  }
  if (
    result.outcome === 'timeout' ||
    /RPC_INIT_TIMEOUT|RPCInitTimeoutError|RPC initialization timed out/i.test(diagnostics)
  ) {
    return 'The SDK worker did not complete its startup handshake. Re-run with --verbose to inspect the bounded worker output.'
  }
  if (result.outcome === 'spawn-error') {
    return 'The isolated Node.js probe could not be started. Re-run with --verbose for the operating-system error.'
  }
  if (result.outcome === 'protocol-error') {
    return 'The isolated probe exited without a valid result. Re-run with --verbose to inspect its bounded output.'
  }
  if (result.phase === 'close') {
    return 'The SDK worker responded, but its cleanup failed. Re-run with --verbose to inspect the cleanup error.'
  }
  return 'The SDK worker failed its heartbeat. Re-run with --verbose to inspect the bounded worker output.'
}

function formatFailureValue(result: SdkRuntimeProbeResult): string {
  if (result.outcome === 'timeout') return `timed out after ${result.durationMs} ms`
  if (result.signal !== null) return `terminated by ${result.signal}`
  if (result.outcome === 'spawn-error') return 'probe could not start'
  if (result.outcome === 'protocol-error') return 'invalid probe result'
  if (result.phase !== undefined) return `${result.phase} failed`
  return `exited with code ${result.exitCode ?? 'unknown'}`
}

export async function checkSdkRuntime(projectRoot: string): Promise<CheckResult> {
  let entrypoint: string
  try {
    entrypoint = resolveSdkEntrypoint(projectRoot)
  } catch (error) {
    const detail = error instanceof Error ? (error.stack ?? error.message) : String(error)
    return {
      id: 'sdk-runtime',
      label: '@qvac/sdk worker heartbeat',
      status: 'fail',
      severity: 'required',
      value: 'SDK entrypoint not found',
      hint: 'Install @qvac/sdk in this project, or repair the installation, before running --deep.',
      detail: `SDK resolution error:\n${detail}`
    }
  }

  const result = await probeSdkRuntime(entrypoint, projectRoot)
  if (result.outcome === 'pass') {
    return {
      id: 'sdk-runtime',
      label: '@qvac/sdk worker heartbeat',
      status: 'pass',
      severity: 'required',
      value: `${result.durationMs} ms`
    }
  }

  const detail = formatDiagnostics(result)
  return {
    id: 'sdk-runtime',
    label: '@qvac/sdk worker heartbeat',
    status: 'fail',
    severity: 'required',
    value: formatFailureValue(result),
    hint: classifySdkRuntimeFailure(result),
    ...(detail !== undefined ? { detail } : {})
  }
}

export async function collectDeepCheckSection(projectRoot: string): Promise<CheckSection> {
  return {
    id: 'deep',
    title: 'SDK runtime (deep)',
    checks: [await checkSdkRuntime(projectRoot)]
  }
}
