import type { CompletionStats } from '@/schemas'
import type { LlmStats } from '@/server/bare/types/addon-responses'

function finiteNumber(value: number | undefined) {
  return value !== undefined && Number.isFinite(value) ? value : undefined
}

export function normalizeCompletionStats(stats: LlmStats | undefined) {
  if (!stats) return undefined

  const timeToFirstToken = finiteNumber(stats.TTFT)
  const tokensPerSecond = finiteNumber(stats.TPS)
  const cacheTokens = finiteNumber(stats.CacheTokens)
  const promptTokens = finiteNumber(stats.promptTokens)
  const generatedTokens = finiteNumber(stats.generatedTokens)
  const avgConcurrentSeq = finiteNumber(stats.avgConcurrentSeq)

  const normalized: CompletionStats = {
    ...(timeToFirstToken !== undefined && { timeToFirstToken }),
    ...(tokensPerSecond !== undefined && { tokensPerSecond }),
    ...(cacheTokens !== undefined && { cacheTokens }),
    ...(promptTokens !== undefined && { promptTokens }),
    ...(generatedTokens !== undefined && { generatedTokens }),
    ...(avgConcurrentSeq !== undefined && { avgConcurrentSeq }),
    ...(stats.backendDevice !== undefined && { backendDevice: stats.backendDevice })
  }

  if (
    timeToFirstToken === undefined &&
    tokensPerSecond === undefined &&
    cacheTokens === undefined &&
    promptTokens === undefined &&
    generatedTokens === undefined &&
    avgConcurrentSeq === undefined &&
    stats.backendDevice === undefined
  ) {
    return undefined
  }

  return normalized
}

export function stoppedByLength({
  cancelled,
  effectivePredict,
  generatedTokens,
  stoppedAtContextBoundary
}: {
  cancelled: boolean
  effectivePredict: number | undefined
  generatedTokens: number | undefined
  stoppedAtContextBoundary: boolean
}): boolean {
  if (cancelled) return false
  if (stoppedAtContextBoundary) return true
  return (
    effectivePredict !== undefined &&
    effectivePredict > 0 &&
    generatedTokens !== undefined &&
    generatedTokens >= effectivePredict
  )
}
